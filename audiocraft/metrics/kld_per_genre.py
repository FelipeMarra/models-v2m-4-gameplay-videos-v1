# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import logging
import argparse
import contextlib
import typing as tp
from functools import partial

from tqdm import tqdm
import pandas as pd

import torch
import torchmetrics
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_read, audio_info

logger = logging.getLogger(__name__)


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
    #print("@@@@ genre_kld.kl_divergence")
    #print(f"pred_probs: {pred_probs.shape} | target_probs: {target_probs.shape}")
    kl_div = torch.nn.functional.kl_div((pred_probs + epsilon).log(), target_probs, reduction="none")
    kl_div_sum = kl_div.sum(-1)
    #print(f"kl_div_sum: {kl_div_sum.shape}")
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
            self.weight += torch.tensor(kld_scores.size(0), device=self.weight.device)

    def compute(self) -> dict:
        """Computes KL-Divergence across all evaluated pred/target pairs."""
        weight: float = float(self.weight.item())  # type: ignore
        assert weight > 0, "Unable to compute with total number of comparisons <= 0"
        logger.info(f"Computing KL divergence on a total of {weight} samples")
        kld_pq = self.kld_pq_sum.item() / weight  # type: ignore
        kld_qp = self.kld_qp_sum.item() / weight  # type: ignore
        kld_both = kld_pq + kld_qp
        return {'kld': kld_pq, 'kld_pq': kld_pq, 'kld_qp': kld_qp, 'kld_both': kld_both}


class PasstKLDivergenceMetric(KLDivergenceMetric):
    """KL-Divergence metric based on pre-trained PASST classifier on AudioSet.

    From: PaSST: Efficient Training of Audio Transformers with Patchout
    Paper: https://arxiv.org/abs/2110.05069
    Implementation: https://github.com/kkoutini/PaSST

    Follow instructions from the github repo:
    ```
    pip install 'git+https://github.com/kkoutini/passt_hear21@0.0.19#egg=hear21passt'
    ```

    Args:
        pretrained_length (float, optional): Audio duration used for the pretrained model.
    """
    def __init__(self, pretrained_length: tp.Optional[float] = None):
        super().__init__()
        self._initialize_model(pretrained_length)

    def _initialize_model(self, pretrained_length: tp.Optional[float] = None):
        """Initialize underlying PaSST audio classifier."""
        model, sr, max_frames, min_frames = self._load_base_model(pretrained_length)
        self.min_input_frames = min_frames
        self.max_input_frames = max_frames
        self.model_sample_rate = sr
        self.model = model
        self.model.eval()
        self.model.to(self.device)

    def _load_base_model(self, pretrained_length: tp.Optional[float]):
        """Load pretrained model from PaSST."""
        try:
            if pretrained_length == 30:
                from hear21passt.base30sec import get_basic_model  # type: ignore
                max_duration = 30
            elif pretrained_length == 20:
                from hear21passt.base20sec import get_basic_model  # type: ignore
                max_duration = 20
            else:
                from hear21passt.base import get_basic_model  # type: ignore
                # Original PASST was trained on AudioSet with 10s-long audio samples
                max_duration = 10
            min_duration = 0.15
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install hear21passt to compute KL divergence: ",
                "pip install 'git+https://github.com/kkoutini/passt_hear21@0.0.19#egg=hear21passt'"
            )
        model_sample_rate = 32_000
        max_input_frames = int(max_duration * model_sample_rate)
        min_input_frames = int(min_duration * model_sample_rate)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            model = get_basic_model(mode='logits')
        return model, model_sample_rate, max_input_frames, min_input_frames

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
        """Run the pretrained model and get the predictions."""
        assert wav.dim() == 3, f"Unexpected number of dims for preprocessed wav: {wav.shape}"
        wav = wav.mean(dim=1)
        # PaSST is printing a lot of garbage that we are not interested in
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            with torch.no_grad(), _patch_passt_stft():
                logits = self.model(wav.to(self.device))
                probs = torch.softmax(logits, dim=-1)
                return probs

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

# Code for running the metric in a standalone fashion for each genre
GENRES = ['Action', 'Adventure', 'Fighting', 'Platform', 'Puzzle', 'RPG', 'Racing', 'Shooters', 'Simulation', 'Sports', 'Strategy']

class GenreAudioDataset(Dataset):
    def __init__(self, data_rows, model_sr, duration=30):
        self.data_rows = data_rows
        self.model_sr = model_sr
        self.duration = duration
        self.n_frames = duration * model_sr

    def __len__(self):
        return len(self.data_rows)

    def set_n_frames(self, x):
        if x.shape[-1] < self.n_frames:
            pad = self.n_frames - x.shape[-1]
            return F.pad(x, (0,pad), mode="constant", value=0)

        if x.shape[-1] > self.n_frames:
            return x[..., :self.n_frames]

        return x

    def __getitem__(self, idx):
        y_pred_path, y_path, y_seek, _ = self.data_rows[idx]

        # Load and convert original audio
        y_out, y_sr = audio_read(y_path, y_seek, duration=30)
        y_out = convert_audio(y_out, y_sr, self.model_sr, 1)
        y_out = self.set_n_frames(y_out)        

        # Load and convert predicted audio
        y_pred_out, y_pred_sr = audio_read(y_pred_path + '.wav')
        y_pred_out = convert_audio(y_pred_out, y_pred_sr, self.model_sr, 1)
        y_pred_out = self.set_n_frames(y_pred_out)        
        
        y_pred_n_frames = y_pred_out.shape[-1]

        return {
            'y_pred': y_pred_out,
            'y': y_out,
            'n_frames': y_pred_n_frames,
            'sr': self.model_sr
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--games_list', type=str, required=False, help="comma-separated list of games")
    parser.add_argument('--eval_path', type=str, required=True, help="path for the eval xp folder. /eval_gen/pred_to_orig.csv are automatically added")
    parser.add_argument('--dataset_path', type=str, default="/home/es119256/dados/datasets/vmdb/nintendo-snes-spc", help="path for the snes mvdb dataset games folder")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size for evaluation")
    parser.add_argument('--num_workers', type=int, default=4, help="number of workers for dataloader")
    args = parser.parse_args()

    pred_to_orig_csv_path = os.path.join(args.eval_path, 'eval_gen/pred_to_orig.csv')
    pred_to_orig_df = pd.read_csv(pred_to_orig_csv_path)

    pred_to_orig_dict = {genre: [] for genre in GENRES}

    # Populate dict with y_pred_path, y_path, y_seek and json_path sorted by genre
    for row in tqdm(pred_to_orig_df.itertuples(index=False, name=None), total=len(pred_to_orig_df), desc="Mapping genres"):
        y_pred_path, y_path, y_seek, json_path = row

        with open(json_path, 'r') as f:
            game_genres = json.load(f)["game_genres"]

        for genre in game_genres:
            if genre in pred_to_orig_dict:
                pred_to_orig_dict[genre].append(row)

    # Run metric for each genre
    kl_div_metric = PasstKLDivergenceMetric(pretrained_length=30).cuda()
    model_sr = kl_div_metric.model_sample_rate
    print(f"Model device: {next(kl_div_metric.model.parameters()).device} | sr {model_sr}")

    for genre, items in pred_to_orig_dict.items():

        dataset = GenreAudioDataset(items, model_sr)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            drop_last=False
        )

        for batch in tqdm(dataloader, desc=f"KLD for genre {genre}"):
            batch_y_pred = batch['y_pred'].cuda()
            batch_y = batch['y'].cuda()
            batch_n_frames = batch['n_frames']
            batch_sr = batch['sr']

            kl_div_metric.update(batch_y_pred, batch_y, batch_n_frames, batch_sr)

        metric_value = kl_div_metric.compute()
        print(f"Passt KL Divergence Metric {genre}: {metric_value}\n")
        logger.info(f"Passt KL Divergence Metric for {genre}: {metric_value}\n")
        
        kl_div_metric.reset()