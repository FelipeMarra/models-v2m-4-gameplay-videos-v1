import os
import logging
import contextlib
import typing as tp
from functools import partial

from ..environment import AudioCraftEnvironment

import torch
import torchmetrics
import torch.nn as nn

from ..data.audio_utils import convert_audio

logger = logging.getLogger(__name__)

GENRES = ['Action', 'Adventure', 'Fighting', 'Platform', 'Puzzle', 'RPG', 'Racing', 'Shooters', 'Simulation', 'Sports', 'Strategy']

class _patch_passt_stft:
    """Decorator to patch torch.stft in PaSST."""
    def __init__(self):
        self.old_stft = torch.stft

    def __enter__(self):
        # return_complex is a mandatory parameter in latest torch versions
        # torch is throwing RuntimeErrors when not set
        torch.stft = partial(torch.stft, return_complex=False)

    def __exit__(self, *exc):
        torch.stft = self.old_stft

class PaSSTMVDB(nn.Module):
    """
    From:https://github.com/FelipeMarra/passt-on-vmdb
    """
    def __init__(self):
        try:
            from hear21passt.base import get_basic_model, get_model_passt
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install hear21passt to compute KL divergence: ",
                "pip install 'git+https://github.com/kkoutini/passt_hear21@0.0.19#egg=hear21passt'"
            )

        super(PaSSTMVDB, self).__init__()

        # models are trained on 10 seconds audios from Audioset, but accept longer audios (20s, or 30s)
        # These models are trained by sampling a 10-second time-pos-encodings sequence 
        self.passt = get_basic_model(mode="logits")
        self.passt.net = get_model_passt("passt_30sec", input_tdim=3000)

        self.class_head = nn.Linear(527, 1)

    def forward(self, x):
        logits = self.passt(x)
        logits = self.class_head(logits)
        logits = nn.Sigmoid()(logits)

        return logits

def kl_divergence(pred_probs: torch.Tensor, target_probs: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Computes the elementwise KL-Divergence loss between probability distributions
    from generated samples and target samples.

    Args:
        pred_probs (torch.Tensor): Probabilities for each label obtained
            from a classifier on generated audio. Expected shape is [B, num_classes].
        target_probs (torch.Tensor): Probabilities for each label obtained
            from a classifier on target audio. Expected shape is [B, num_classes].
        epsilon (float): Epsilon value.
    Returns:
        kld (torch.Tensor): KLD loss between each generated sample and target pair.
    """
    kl_div = torch.nn.functional.kl_div((pred_probs + epsilon).log(), target_probs, reduction="none")
    kl_div_sum = kl_div.sum(-1)

    return kl_div_sum

class KLDivergenceMetric(torchmetrics.Metric):
    """Base implementation for KL Divergence metric.

    The KL divergence is measured between probability distributions
    of class predictions returned by a pre-trained audio classification model.
    When the KL-divergence is low, the generated audio is expected to
    have similar acoustic characteristics as the reference audio,
    according to the classifier.
    """
    def __init__(self):
        super().__init__()
        self.add_state("kld_pq_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("kld_qp_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("kld_all_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0), dist_reduce_fx="sum")

    def _get_label_distribution(self, x: torch.Tensor, sizes: torch.Tensor,
                                sample_rates: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """Get model output given provided input tensor.

        Args:
            x (torch.Tensor): Input audio tensor of shape [B, C, T].
            sizes (torch.Tensor): Actual audio sample length, of shape [B].
            sample_rates (torch.Tensor): Actual audio sample rate, of shape [B].
        Returns:
            probs (torch.Tensor): Probabilities over labels, of shape [B, num_classes].
        """
        raise NotImplementedError("implement method to extract label distributions from the model.")

    def update(self, preds: torch.Tensor, targets: torch.Tensor,
                sizes: torch.Tensor, sample_rates: torch.Tensor) -> None:
        """Calculates running KL-Divergence loss between batches of audio
        preds (generated) and target (ground-truth)
        Args:
            preds (torch.Tensor): Audio samples to evaluate, of shape [B, C, T].
            targets (torch.Tensor): Target samples to compare against, of shape [B, C, T].
            sizes (torch.Tensor): Actual audio sample length, of shape [B].
            sample_rates (torch.Tensor): Actual audio sample rate, of shape [B].
        """
        assert preds.shape == targets.shape
        assert preds.size(0) > 0, "Cannot update the loss with empty tensors"
        preds_probs = self._get_label_distribution(preds, sizes, sample_rates)
        targets_probs = self._get_label_distribution(targets, sizes, sample_rates)
        if preds_probs is not None and targets_probs is not None:
            assert preds_probs.shape == targets_probs.shape
            kld_scores = kl_divergence(preds_probs, targets_probs)
            assert not torch.isnan(kld_scores).any(), "kld_scores contains NaN value(s)!"
            self.kld_pq_sum += torch.sum(kld_scores)
            kld_qp_scores = kl_divergence(targets_probs, preds_probs)
            self.kld_qp_sum += torch.sum(kld_qp_scores)
            self.weight += torch.tensor(kld_scores.size(0))

    def compute(self) -> dict:
        """Computes KL-Divergence across all evaluated pred/target pairs."""
        weight: float = float(self.weight.item())  # type: ignore
        assert weight > 0, "Unable to compute with total number of comparisons <= 0"
        logger.info(f"Computing KL divergence on a total of {weight} samples")
        kld_pq = self.kld_pq_sum.item() / weight  # type: ignore
        kld_qp = self.kld_qp_sum.item() / weight  # type: ignore
        kld_both = kld_pq + kld_qp
        return {'genre_kld': kld_pq, 'genre_kld_pq': kld_pq, 'genre_kld_qp': kld_qp, 'genre_kld_both': kld_both}

class GenreKLDivergenceMetric(KLDivergenceMetric):
    """KL-Divergence metric based on tuned and modified PASST classifier on the VMDB dataset

    Based on the PasstKLDivergenceMetric class

    The weights of the Genre Classifier are expected at the `genre_classifier` folder inside the reference dir (one GENRE/checkpoint/best_model.pth for each GENRE in the classifier)
    """
    def __init__(self, checkpoints_path):
        assert checkpoints_path != None, "metrics.genre_kld.checkpoints must be set to a path containing a checkpoint for each genre"
        super().__init__()
        self.checkpoints_path = AudioCraftEnvironment.resolve_reference_path(checkpoints_path)
        self._initialize_model()

    def _initialize_model(self):
        """Initialize underlying PaSST audio classifier."""
        model, sr, max_frames, min_frames = self._load_base_model()
        self.min_input_frames = min_frames
        self.max_input_frames = max_frames
        self.model_sample_rate = sr
        self.model = model
        self.model.eval()
        self.model.to(self.device)

    def _load_base_model(self):
        """Load pretrained model from PaSST."""
        max_duration = 30
        min_duration = 0.15
        model_sample_rate = 32_000
        max_input_frames = int(max_duration * model_sample_rate)
        min_input_frames = int(min_duration * model_sample_rate)

        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f): # supress PaSST prints
            model = PaSSTMVDB()

        return model, model_sample_rate, max_input_frames, min_input_frames

    def _load_genre_checkpoint(self, ckpt_path:str):
        state_dict = torch.load(ckpt_path)
        self.model.load_state_dict(state_dict['model_sate'])
        self.model.cuda()

    def _process_audio(self, wav: torch.Tensor, sample_rate: int, wav_len: int) -> tp.List[torch.Tensor]:
        """Process audio to feed to the pretrained model."""
        wav = wav.unsqueeze(0)
        wav = wav[..., :wav_len]
        wav = convert_audio(wav, from_rate=sample_rate, to_rate=self.model_sample_rate, to_channels=1)
        wav = wav.squeeze(0)
        # we don't pad but return a list of audio segments as this otherwise affects the KLD computation
        segments = torch.split(wav, self.max_input_frames, dim=-1)
        valid_segments = []
        for s in segments:
            # ignoring too small segments that are breaking the model inference
            if s.size(-1) > self.min_input_frames:
                valid_segments.append(s)
        return [s[None] for s in valid_segments]

    def _get_model_preds(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Run the pretrained model and get the predictions.

        Args:
            wav (torch.Tensor): Audio samples to evaluate, of shape [B, C, T]. They must be 32KHz.

        Return:
            Tensor with shape (B, G), where G is the amount of genres and B is the batch size
        """
        assert wav.dim() == 3, f"Unexpected number of dims for preprocessed wav: {wav.shape}"
        wav = wav.mean(dim=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f): # Supress PaSST prints
            with torch.no_grad(), _patch_passt_stft():
                probs_batch = None # type: ignore

                for genre in GENRES:
                    ckpt_path = os.path.join(self.checkpoints_path, genre, 'checkpoints', 'best_model.pth')
                    self._load_genre_checkpoint(ckpt_path)

                    res:torch.Tensor = self.model(wav.to(self.device)) # (B, 1)

                    if isinstance(probs_batch, torch.Tensor):
                        probs_batch = torch.cat((probs_batch, res), dim=1) # (B, G)
                    else:
                        probs_batch:torch.Tensor = res # (B, 1)

                return probs_batch

    def _get_label_distribution(self, x: torch.Tensor, sizes: torch.Tensor,
                                sample_rates: torch.Tensor) -> tp.Optional[torch.Tensor]:
        """Get model output given provided input tensor.

        Args:
            x (torch.Tensor): Input audio tensor of shape [B, C, T].
            sizes (torch.Tensor): Actual audio sample length, of shape [B].
            sample_rates (torch.Tensor): Actual audio sample rate, of shape [B].
        Returns:
            probs (torch.Tensor, optional): Probabilities over labels, of shape [B, num_classes].
        """
        all_probs: tp.List[torch.Tensor] = []
        for i, wav in enumerate(x):
            sample_rate = int(sample_rates[i].item())
            wav_len = int(sizes[i].item())
            wav_segments = self._process_audio(wav, sample_rate, wav_len)
            for segment in wav_segments:
                probs = self._get_model_preds(segment).mean(dim=0)
                all_probs.append(probs)
        if len(all_probs) > 0:
            return torch.stack(all_probs, dim=0)
        else:
            return None

