import os
import json
import logging
import argparse
import contextlib
import typing as tp
from functools import partial
#from ..environment import AudioCraftEnvironment

import pandas as pd
from tqdm import tqdm

import torch
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, F1Score
import torch.nn as nn

logger = logging.getLogger(__name__)

GENRES = ['Action', 'Adventure', 'Fighting', 'Platform', 'Puzzle', 'RPG', 'Racing', 'Shooters', 'Simulation', 'Sports', 'Strategy']

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

class ImgBindOnMVDB(nn.Module):
    def __init__(self):
        super(ImgBindOnMVDB, self).__init__()

        self.img_bind = imagebind_model.imagebind_huge(pretrained=True)

        self.lin_1 = nn.Linear(1024, 1024)
        self.lin_2 = nn.Linear(1024, 512)
        self.lin_3 = nn.Linear(512, 256)
        self.class_head = nn.Linear(256, 1)

    def forward(self, audios_paths):
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audios_paths, "cuda")
        }

        with torch.no_grad():
            embeddings = self.img_bind(inputs)[ModalityType.AUDIO]

        logits = self.lin_1(embeddings)
        logits = nn.GELU()(logits)

        logits = self.lin_2(logits)
        logits = nn.GELU()(logits)

        logits = self.lin_3(logits)
        logits = nn.GELU()(logits)

        logits = self.class_head(logits)
        logits = nn.Sigmoid()(logits)

        return logits

class GenreClassificationMetrics(torchmetrics.Metric):
    """Base implementation for Genre Classifications metrics.
    """
    def __init__(self):
        super().__init__()

        self.metrics = MetricCollection([
            Accuracy(task='multilabel', average='none', num_labels=len(GENRES)).cuda(),
            F1Score(task='multilabel', average='none', num_labels=len(GENRES)).cuda()
        ])

    def _get_label_distribution(self, x: list[str]) -> tp.Optional[torch.Tensor]:
        """Get model output given provided input tensor.

        Args:
            x (torch.Tensor): Input audio tensor of shape [B, C, T].
            sizes (torch.Tensor): Actual audio sample length, of shape [B].
            sample_rates (torch.Tensor): Actual audio sample rate, of shape [B].
        Returns:
            probs (torch.Tensor): Probabilities over labels, of shape [B, num_classes].
        """
        raise NotImplementedError("implement method to extract label distributions from the model.")

    def update(self, preds:list[str], jsons_paths:list[str]) -> None:
        """Calculates running KL-Divergence loss between batches of audio
        preds (generated) and target (ground-truth)
        Args:
            preds (torch.Tensor): Audio samples to evaluate, of shape [B, C, T].
            targets (torch.Tensor): Target samples to compare against, of shape [B, C, T].
            sizes (torch.Tensor): Actual audio sample length, of shape [B].
            sample_rates (torch.Tensor): Actual audio sample rate, of shape [B].
        """
        assert len(preds) > 0, "Cannot update the loss with empty tensors"
        preds_probs = self._get_label_distribution(preds) # (B, G)

        # Get gorund truth labels from json
        tgt_labels = []

        for json_path in jsons_paths:
            with open(json_path, 'r') as f:
                gt_labels = json.load(f)["game_genres"]
            gt_labels = [1 if g in gt_labels else 0 for g in GENRES]
            tgt_labels.append(torch.Tensor(gt_labels))

        tgt_labels = torch.stack(tgt_labels, dim=0).cuda()

        if preds_probs is not None and tgt_labels is not None:
            assert preds_probs.shape == tgt_labels.shape
            self.metrics.update(preds_probs, tgt_labels)

    def compute(self) -> dict:
        """Computes metrics in `self.metrics` across all evaluated pred/target pairs."""
        logger.info(f"Computing {self.metrics.keys()} on a total of TODO samples")

        comp_metrics = self.metrics.compute()
        genre_comp_metrics = {}

        for metric in comp_metrics:
            for metric_value, genre in zip(comp_metrics[metric], GENRES):
                genre_comp_metrics[f'{metric}_{genre}'] = metric_value

        return genre_comp_metrics

class ImgBindGenreClassificationMetric(GenreClassificationMetrics):
    """Classification metrics based on tuned and modified PASST classifier on the VMDB dataset

    Based on the PasstKLDivergenceMetric class

    The weights of the Genre Classifier are expected at the `genre_classifier` folder inside the reference dir (one GENRE/checkpoint/best_model.pth for each GENRE in the classifier)
    """
    def __init__(self, checkpoints_path):
        assert checkpoints_path != None, "metrics.genre_kld.checkpoints must be set to a path containing a checkpoint for each genre"
        super().__init__()
        #self.checkpoints_path = AudioCraftEnvironment.resolve_reference_path(checkpoints_path)
        self.checkpoints_path = checkpoints_path
        self._initialize_model()

    def _initialize_model(self):
        """Initialize underlying PaSST audio classifier."""
        model = self._load_base_model()
        self.model = model
        self.model.eval()
        self.model.cuda()

    def _load_base_model(self):
        """Load pretrained model from Image Bind."""
        model = ImgBindOnMVDB()
        return model

    def _load_genre_checkpoint(self, ckpt_path:str):
        state_dict = torch.load(ckpt_path)
        self.model.load_state_dict(state_dict['model_sate'])
        self.model.eval()
        self.model.cuda()

    def _get_model_preds(self, wav_path: list[str]) -> torch.Tensor:
        """
        Run the pretrained model and get the predictions.

        Args:
            wav (torch.Tensor): Audio samples to evaluate, of shape [B, C, T]. They must be 32KHz.

        Return:
            Tensor with shape (B, G), where G is the amount of genres and B is the batch size
        """
        with torch.no_grad():
            probs_batch = None # type: ignore

            for genre in GENRES:
                ckpt_path = os.path.join(self.checkpoints_path, genre, 'checkpoints', 'best_model.pth')
                self._load_genre_checkpoint(ckpt_path)

                res:torch.Tensor = self.model(wav_path) # (B, 1)

                if isinstance(probs_batch, torch.Tensor):
                    probs_batch = torch.cat((probs_batch, res), dim=1) # (B, G)
                else:
                    probs_batch:torch.Tensor = res # (B, 1)

            return probs_batch

    def _get_label_distribution(self, x: list[str]) -> tp.Optional[torch.Tensor]:
        """Get model output given provided input tensor.

        Args:
            x (torch.Tensor): Input audio tensor of shape [B, C, T].
            sizes (torch.Tensor): Actual audio sample length, of shape [B].
            sample_rates (torch.Tensor): Actual audio sample rate, of shape [B].
        Returns:
            probs (torch.Tensor, optional): Probabilities over labels, of shape [B, num_classes].
        """
        probs = self._get_model_preds(x)
        return probs 

def run_genre_acc_standalone(checkpoints_path, df):
    """
        Standalone ImageBind Score for csv generated by enabeling evaluate.metrics.save_eval_gen
    """
    img_bind_cls = ImgBindGenreClassificationMetric(checkpoints_path).cuda()
    batch_size = 32
    batch_y_pred = []
    batch_json_path = []

    for row in tqdm(df.itertuples(index=False, name=None), total=len(df)):
        y_pred_path, y_path, y_seek, json_path = row

        if len(batch_y_pred)-1 < batch_size:
            batch_y_pred.append(y_pred_path+'.wav')
            batch_json_path.append(json_path)
        else:
            img_bind_cls.update(batch_y_pred, batch_json_path)

            batch_y_pred = [y_pred_path+'.wav']
            batch_json_path = [json_path]

    metric_value = img_bind_cls.compute()

    print(f"\nImageBind Score: {metric_value}\n")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints_path', type=str, required=False)
    parser.add_argument('--eval_path', type=str, required=True, help="path for the eval xp folder. /eval_gen/pred_to_orig.csv are automatically added")
    parser.add_argument('--dataset_path', type=str, default="/home/es119256/dados/datasets/vmdb/nintendo-snes-spc", help="path for the snes mvdb dataset games folder")
    args = parser.parse_args()

    csv_path = os.path.join(args.eval_path, 'eval_gen/pred_to_orig.csv')
    df = pd.read_csv(csv_path)

    run_genre_acc_standalone(args.checkpoints_path, df)