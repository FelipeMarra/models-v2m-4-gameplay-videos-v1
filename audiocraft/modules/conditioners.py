# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
import logging
import random
import re
import typing as tp
import warnings
import einops
from num2words import num2words
import spacy
from transformers import T5EncoderModel, T5Tokenizer  # type: ignore
from transformers import VivitModel
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchvision
from torchvision.transforms import v2
from .streaming import StreamingModule
from .transformer import create_sin_embedding
from ..data.audio_dataset import SegmentInfo
from ..data.audio_utils import convert_audio
from ..utils.autocast import TorchAutocast
from ..quantization import ResidualVectorQuantizer
from ..utils.utils import collate, hash_trick, length_to_mask
import av
import numpy as np

logger = logging.getLogger(__name__)
TextCondition = tp.Optional[str]  # a text condition can be a string or None (if doesn't exist)
ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask


class WavCondition(tp.NamedTuple):
    wav: torch.Tensor
    length: torch.Tensor
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


class JointEmbedCondition(tp.NamedTuple):
    wav: torch.Tensor
    text: tp.List[tp.Optional[str]]
    video: tp.List[tp.Optional[str]]
    length: torch.Tensor
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


class SymbolicCondition(tp.NamedTuple):
    frame_chords: tp.Optional[torch.Tensor] = None
    melody: tp.Optional[torch.Tensor] = None


@dataclass
class ConditioningAttributes:
    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    video: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    wav: tp.Dict[str, WavCondition] = field(default_factory=dict)
    joint_embed: tp.Dict[str, JointEmbedCondition] = field(default_factory=dict)
    symbolic: tp.Dict[str, SymbolicCondition] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def video_attributes(self):
        return self.video.keys()

    @property
    def wav_attributes(self):
        return self.wav.keys()

    @property
    def joint_embed_attributes(self):
        return self.joint_embed.keys()

    @property
    def symbolic_attributes(self):
        return self.symbolic.keys()

    @property
    def attributes(self):
        return {
            "text": self.text_attributes,
            "wav": self.wav_attributes,
            "video": self.video_attributes,
            "joint_embed": self.joint_embed_attributes,
            "symbolic": self.symbolic_attributes,
        }

    def to_flat_dict(self):
        return {
            **{f"text.{k}": v for k, v in self.text.items()},
            **{f"video.{k}": v for k, v in self.video.items()},
            **{f"wav.{k}": v for k, v in self.wav.items()},
            **{f"joint_embed.{k}": v for k, v in self.joint_embed.items()},
            **{f"symbolic.{k}": v for k, v in self.symbolic.items()}
        }

    @classmethod
    def from_flat_dict(cls, x):
        out = cls()
        for k, v in x.items():
            kind, att = k.split(".")
            out[kind][att] = v
        return out


class SegmentWithAttributes(SegmentInfo):
    """Base class for all dataclasses that are used for conditioning.
    All child classes should implement `to_condition_attributes` that converts
    the existing attributes to a dataclass of type ConditioningAttributes.
    """
    def to_condition_attributes(self) -> ConditioningAttributes:
        raise NotImplementedError()


def nullify_condition(condition: ConditionType, dim: int = 1):
    """Transform an input condition to a null condition.
    The way it is done by converting it to a single zero vector similarly
    to how it is done inside WhiteSpaceTokenizer and NoopTokenizer.

    Args:
        condition (ConditionType): A tuple of condition and mask (tuple[torch.Tensor, torch.Tensor])
        dim (int): The dimension that will be truncated (should be the time dimension)
        WARNING!: dim should not be the batch dimension!
    Returns:
        ConditionType: A tuple of null condition and mask
    """
    assert dim != 0, "dim cannot be the batch dimension!"
    assert isinstance(condition, tuple) and \
        isinstance(condition[0], torch.Tensor) and \
        isinstance(condition[1], torch.Tensor), "'nullify_condition' got an unexpected input type!"
    cond, mask = condition
    B = cond.shape[0]
    last_dim = cond.dim() - 1
    out = cond.transpose(dim, last_dim)
    out = 0. * out[..., :1]
    out = out.transpose(dim, last_dim)
    mask = torch.zeros((B, 1), device=out.device).int()
    assert cond.dim() == out.dim()
    return out, mask


def nullify_wav(cond: WavCondition) -> WavCondition:
    """Transform a WavCondition to a nullified WavCondition.
    It replaces the wav by a null tensor, forces its length to 0, and replaces metadata by dummy attributes.

    Args:
        cond (WavCondition): Wav condition with wav, tensor of shape [B, T].
    Returns:
        WavCondition: Nullified wav condition.
    """
    null_wav, _ = nullify_condition((cond.wav, torch.zeros_like(cond.wav)), dim=cond.wav.dim() - 1)
    return WavCondition(
        wav=null_wav,
        length=torch.tensor([0] * cond.wav.shape[0], device=cond.wav.device),
        sample_rate=cond.sample_rate,
        path=[None] * cond.wav.shape[0],
        seek_time=[None] * cond.wav.shape[0],
    )


def nullify_joint_embed(embed: JointEmbedCondition) -> JointEmbedCondition:
    """Nullify the joint embedding condition by replacing it by a null tensor, forcing its length to 0,
    and replacing metadata by dummy attributes.

    Args:
        cond (JointEmbedCondition): Joint embedding condition with wav and text, wav tensor of shape [B, C, T].
    """
    null_wav, _ = nullify_condition((embed.wav, torch.zeros_like(embed.wav)), dim=embed.wav.dim() - 1)
    return JointEmbedCondition(
        wav=null_wav, 
        text=[None] * len(embed.text),
        video=[None] * len(embed.video),
        length=torch.LongTensor([0]).to(embed.wav.device),
        sample_rate=embed.sample_rate,
        path=[None] * embed.wav.shape[0],
        seek_time=[0] * embed.wav.shape[0],
    )

def _drop_description_condition(conditions: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
    """Drop the text condition but keep the wav conditon on a list of ConditioningAttributes.
    This is useful to calculate l_style in the double classifier free guidance formula.
    See paragraph 4.3 in https://arxiv.org/pdf/2407.12563

    Args:
        conditions (tp.List[ConditioningAttributes]): List of conditions.
    """
    # We assert that description and self_wav are in the conditions
    for condition in conditions:
        assert 'description' in condition.text.keys()
        assert 'self_wav' in condition.wav.keys()
    return AttributeDropout(p={'text': {'description': 1.0},
                               'wav': {'self_wav': 0.0}})(conditions)

class Tokenizer:
    """Base tokenizer implementation
    (in case we want to introduce more advances tokenizers in the future).
    """
    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class WhiteSpaceTokenizer(Tokenizer):
    """This tokenizer should be used for natural language descriptions.
    For example:
    ["he didn't, know he's going home.", 'shorter sentence'] =>
    [[78, 62, 31,  4, 78, 25, 19, 34],
    [59, 77,  0,  0,  0,  0,  0,  0]]
    """
    PUNCTUATION = "?:!.,;"

    def __init__(self, n_bins: int, pad_idx: int = 0, language: str = "en_core_web_sm",
                 lemma: bool = True, stopwords: bool = True) -> None:
        self.n_bins = n_bins
        self.pad_idx = pad_idx
        self.lemma = lemma
        self.stopwords = stopwords
        try:
            self.nlp = spacy.load(language)
        except IOError:
            spacy.cli.download(language)  # type: ignore
            self.nlp = spacy.load(language)

    @tp.no_type_check
    def __call__(self, texts: tp.List[tp.Optional[str]],
                 return_text: bool = False) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Take a list of strings and convert them to a tensor of indices.

        Args:
            texts (list[str]): List of strings.
            return_text (bool, optional): Whether to return text as additional tuple item. Defaults to False.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Indices of words in the LUT.
                - And a mask indicating where the padding tokens are
        """
        output, lengths = [], []
        texts = deepcopy(texts)
        for i, text in enumerate(texts):
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(torch.Tensor([self.pad_idx]))
                lengths.append(0)
                continue

            # convert numbers to words
            text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)  # type: ignore
            # normalize text
            text = self.nlp(text)  # type: ignore
            # remove stopwords
            if self.stopwords:
                text = [w for w in text if not w.is_stop]  # type: ignore
            # remove punctuation
            text = [w for w in text if w.text not in self.PUNCTUATION]  # type: ignore
            # lemmatize if needed
            text = [getattr(t, "lemma_" if self.lemma else "text") for t in text]  # type: ignore

            texts[i] = " ".join(text)
            lengths.append(len(text))
            # convert to tensor
            tokens = torch.Tensor([hash_trick(w, self.n_bins) for w in text])
            output.append(tokens)

        mask = length_to_mask(torch.IntTensor(lengths)).int()
        padded_output = pad_sequence(output, padding_value=self.pad_idx).int().t()
        if return_text:
            return padded_output, mask, texts  # type: ignore
        return padded_output, mask

class BaseConditioner(nn.Module):
    """Base model for all conditioner modules.
    We allow the output dim to be different than the hidden dim for two reasons:
    1) keep our LUTs small when the vocab is large;
    2) make all condition dims consistent.

    Args:
        dim (int): Hidden dim of the model.
        output_dim (int): Output dim of the conditioner.
    """
    def __init__(self, dim: int, output_dim: int, output_tkns_dim:int=-1, seq_len: int=-1):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.output_tkns_dim = output_tkns_dim
        self.seq_len = seq_len

        if self.output_dim > -1:  # omit projection when output_dim <= 0
            self.output_proj = nn.Linear(self.dim, self.output_dim)

        if self.output_tkns_dim > -1 and self.seq_len > -1:  # omit projection when output_dim <= 0
            self.output_tkns_proj = nn.Linear(self.seq_len, self.output_tkns_dim)

    def apply_output_tkns_proj(self, x:torch.Tensor, name="") -> torch.Tensor:
        if self.output_tkns_proj:
            # B, Seq_len, output_dim
            B, S, O = x.shape
            # print(f"\n {name} apply_output_tkns_proj, x.shape: {x.shape} \n")

            # Put Seq_len in the last dim, apply layer to change num of tokens and reshape back
            x = x.permute(0, 2, 1) # B, S, O (batch, seq_len, out_dim) -> B, O, S
            x = self.output_tkns_proj(x)
            x = x.permute(0, 2, 1) # B, S, O -> B, O, S

        return x

    def tokenize(self, *args, **kwargs) -> tp.Any:
        """Should be any part of the processing that will lead to a synchronization
        point, e.g. BPE tokenization with transfer to the GPU.

        The returned value will be saved and return later when calling forward().
        """
        raise NotImplementedError()

    def forward(self, inputs: tp.Any) -> ConditionType:
        """Gets input that should be used as conditioning (e.g, genre, description or a waveform).
        Outputs a ConditionType, after the input data was embedded as a dense vector.

        Returns:
            ConditionType:
                - A tensor of size [B, T, D] where B is the batch size, T is the length of the
                  output embedding and D is the dimension of the embedding.
                - And a mask indicating where the padding tokens.
        """
        raise NotImplementedError()

class JointEmbeddingConditioner(BaseConditioner):
    """Joint embedding conditioning supporting both audio or text conditioning.

    Args:
        dim (int): Dimension.
        output_dim (int): Output dimension.
        device (str): Device.
        attribute (str): Attribute used by the conditioner.
        autocast_dtype (str): Autocast for the conditioner.
        quantize (bool): Whether to quantize the CLAP embedding.
        n_q (int): Number of residual quantizers (used if quantize is true).
        bins (int): Quantizers' codebooks size (used if quantize is true).
        kwargs: Additional parameters for residual vector quantizer.
    """
    def __init__(self, dim: int, output_dim: int, device: str, attribute: str,
                 autocast_dtype: tp.Optional[str] = 'float32', quantize: bool = True,
                 n_q: int = 12, bins: int = 1024, **kwargs):
        super().__init__(dim=dim, output_dim=output_dim)
        self.device = device
        self.attribute = attribute
        if autocast_dtype is None or device == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
            logger.warning("JointEmbeddingConditioner has no autocast, this might lead to NaN.")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"JointEmbeddingConditioner will be evaluated with autocast as {autocast_dtype}.")
            self.autocast = TorchAutocast(enabled=True, device_type=self.device, dtype=dtype)
        # residual vector quantizer to discretize the conditioned embedding
        self.quantizer: tp.Optional[ResidualVectorQuantizer] = None
        if quantize:
            self.quantizer = ResidualVectorQuantizer(dim, n_q=n_q, bins=bins, **kwargs)

    def _get_embed(self, x: JointEmbedCondition) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Get joint embedding in latent space from the inputs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tensor for the latent embedding
                and corresponding empty indexes.
        """
        raise NotImplementedError()

    def forward(self, x: JointEmbedCondition) -> ConditionType:
        with self.autocast:
            embed, empty_idx = self._get_embed(x)
            if self.quantizer is not None:
                embed = embed.view(-1, self.dim, 1)
                q_res = self.quantizer(embed, frame_rate=1)
                out_embed = q_res.x.view(-1, self.dim)
            else:
                out_embed = embed
            out_embed = self.output_proj(out_embed).view(-1, 1, self.output_dim)
            mask = torch.ones(*out_embed.shape[:2], device=out_embed.device)
            mask[empty_idx, :] = 0  # zero-out index where the input is non-existant
            out_embed = (out_embed * mask.unsqueeze(-1))
            return out_embed, mask

    def tokenize(self, x: JointEmbedCondition) -> JointEmbedCondition:
        return x

class TextConditioner(BaseConditioner):
    ...


class T5Conditioner(TextConditioner):
    """T5-based TextConditioner.

    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        device (str): Device for T5 Conditioner.
        autocast_dtype (tp.Optional[str], optional): Autocast dtype.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
    """
    MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(self, name: str, output_dim: int, finetune: bool, device: str,
                 autocast_dtype: tp.Optional[str] = 'float32', word_dropout: float = 0.,
                 normalize_text: bool = False, output_tkns_dim:int=-1, seq_len: int=-1):
        assert name in self.MODELS, f"Unrecognized t5 model name (should in {self.MODELS})"

        super().__init__(self.MODELS_DIMS[name], output_dim, output_tkns_dim=output_tkns_dim, seq_len=seq_len)
        self.device = device
        self.name = name
        self.finetune = finetune
        self.word_dropout = word_dropout

        if autocast_dtype is None or self.device == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
            if self.device != 'cpu':
                logger.warning("T5 has no autocast, this might lead to NaN")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"T5 will be evaluated with autocast as {autocast_dtype}")
            self.autocast = TorchAutocast(enabled=True, device_type=self.device, dtype=dtype)

        # Let's disable logging temporarily because T5 will vomit some errors otherwise.
        # thanks https://gist.github.com/simon-weber/7853144
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer:T5Tokenizer = T5Tokenizer.from_pretrained(name)
                t5:T5EncoderModel = T5EncoderModel.from_pretrained(name).train(mode=finetune) # type: ignore
            finally:
                logging.disable(previous_level)

        if finetune:
            self.t5 = t5
        else:
            # this makes sure that the t5 models is not part
            # of the saved checkpoint
            self.__dict__['t5'] = t5.to(device)

        self.normalize_text = normalize_text
        if normalize_text:
            self.text_normalizer = WhiteSpaceTokenizer(1, lemma=True, stopwords=True)

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        # if current sample doesn't have a certain attribute, replace with empty string
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]

        if self.normalize_text:
            _, _, entries = self.text_normalizer(entries, return_text=True) # type: ignore

        if self.word_dropout > 0. and self.training:
            new_entries = []
            for entry in entries:
                words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(
            entries, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.seq_len
        ).to(self.device)

        mask = inputs['attention_mask']
        mask[empty_idx, :] = 0  # type: ignore # zero-out index where the input is non-existant
        return inputs # type: ignore

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs['attention_mask']
        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state

        embeds = self.output_proj(embeds.to(self.output_proj.weight))
        # print(f"\n T5 mask shape: {mask.shape} \n")
        embeds = (embeds * mask.unsqueeze(-1))
        embeds = self.apply_output_tkns_proj(embeds, name=self.name)

        return embeds, mask

class SNESViViTImageProcessor():
    def __init__(self, normalize:bool) -> None:
        self.normalize = normalize

        self.compose = v2.Compose([
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
        ])

    def __call__(self, videos:torch.Tensor) -> torch.Tensor:
        videos = self.compose(videos)

        if self.normalize:
            videos = videos / 255

        return videos

class VideoConditioner(BaseConditioner):
    ...

class ViViTConditioner(VideoConditioner):
    """
        ViViT-based Video Conditioner
    """

    MODELS = ["google/vivit-b-16x2-kinetics400"]
    MODELS_DIMS = {
        "google/vivit-b-16x2-kinetics400": 768,
    }

    def __init__(self, name: str, output_dim: int, finetune:bool, device:str, 
                autocast_dtype: tp.Optional[str] = 'float32', video_len:int=32, 
                num_hidden_layers:int=12, num_attention_heads:int=12, 
                output_tkns_dim:int=-1, seq_len: int=-1):
        assert name in self.MODELS, f"Unrecognized ViViT model name (should in {self.MODELS})"

        super().__init__(self.MODELS_DIMS[name], output_dim, output_tkns_dim=output_tkns_dim, seq_len=seq_len)

        self.device = device
        self.name = name
        self.finetune = finetune
        self.video_len = video_len

        if autocast_dtype is None or self.device == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
            if self.device != 'cpu':
                logger.warning("ViViT has no autocast, this might lead to NaN")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"ViViT will be evaluated with autocast as {autocast_dtype}")
            self.autocast = TorchAutocast(enabled=True, device_type=self.device, dtype=dtype)

        self.image_processor = SNESViViTImageProcessor(normalize=True)

        vivit_config = VivitModel.config_class.from_pretrained(name)
        # print(f"ViViT CONFIG INSIDE VIVIT CONDITIONER\n{vivit_config}")

        vivit_config.update(
            {
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": num_attention_heads
            }
        )
        # print(f"ViViT CONFIG INSIDE VIVIT CONDITIONER MODIFIED\n{vivit_config}")

        vivit = VivitModel.from_pretrained(name, config=vivit_config).train(mode=finetune) # type: ignore
        #vivit = VivitModel.from_pretrained(name).train(mode=finetune) # type: ignore

        if finetune:
            self.vivit = vivit
        else:
            # this makes sure that the vivit models is not part of the saved checkpoint
            self.__dict__['vivit'] = vivit.to(device)

    def read_video_pyav(self, container, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`list[int]`): List of frame indices to decode.
        Returns:
            result (torch.Tensor):tensor of decoded frames of shape (num_frames, 3, height, width).
        '''
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        video = torch.stack([torch.from_numpy(x.to_ndarray(format="rgb24")) for x in frames])
        video = video.permute(0, 3, 1, 2)
        return video

    def write_video(self, file_name, frame_rate, video_tensor):
        """
            For debugging
        """
        # Def not sufficient to bring the image back to the original pixel values
        # after the ImageProcessor stuff, but enough to get an ideia if it is working
        video_tensor = video_tensor.permute(0, 2, 3, 1)
        torchvision.io.write_video(f'{file_name}.mp4', video_tensor, frame_rate)

    def sample_frame_indices_random_fr(self, video_path, clip_len, seg_len):
        '''
        Sample a given number of frame indices from the video.
        We set a window of a random size and position and sample frames linearly spaced.
        The window size is set so that the stride is at leat 2, that is, we sample
        at least every 2 frames. 
        Since our videos have 300 frames and ViViT asks for 32, we can sample at most
        every 9 frames.
        Args:
            clip_len (`int`): Total number of frames to sample.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`list[int]`): List of sampled frame indices
        '''
        max_fr = np.ceil(seg_len/clip_len) # amout of frames vary a bit
        min_fr = 2

        try:
            frame_sample_rate = np.random.randint(min_fr, max_fr) # rand between 2 and 9 (for 300 frames)
        except:
            logger.error(f"Video {video_path} has len of only {seg_len} and should be removed from the dataset")
            frame_sample_rate=1

        converted_len = int(clip_len * frame_sample_rate) # 32 * 2-9 -> 64-288
        end_idx = np.random.randint(converted_len, seg_len) # end at rand between 64-288 and 299
        # start from rand end - 64
        start_idx = end_idx - converted_len
        # start from rand end minus 64-288 until rand end -> 
        # range from 0 to 299 lin spaced -> step frame_sample_rate
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        window_duration = (end_idx-start_idx)/30
        return indices, window_duration

    def sample_frame_indices(self, video_path, clip_len, frame_sample_rate, seg_len):
        '''
        Sample a given number of frame indices from the video.
        Args:
            clip_len (`int`): Total number of frames to sample.
            frame_sample_rate (`int`): Sample every n-th frame.
            seg_len (`int`): Maximum allowed index of sample's last frame.
        Returns:
            indices (`list[int]`): List of sampled frame indices
        '''
        try:
            converted_len = int(clip_len * frame_sample_rate)
            end_idx = np.random.randint(converted_len, seg_len) # rand between 64 and 300 (for videos with 300 frames)
        except:
            logger.error(f"Video {video_path} has len of only {seg_len} and should be removed from the dataset")
            converted_len = clip_len
            end_idx = np.random.randint(converted_len, seg_len) # rand between 32 and ??? (for videos with less than 64 frames, that should be remove from the dataset)

        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, tp.List[tp.Any]]:
        # video_len: video total seconds
        # if current sample doesn't have a certain attribute, replace with empty string
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        videos:tp.Dict[str, tp.Any] = {"video": [], "attention_mask": []}
        for v in entries:
            if v != "":
                if isinstance(v, str):
                    container = av.open(v)
                    #indices, window_duration = self.sample_frame_indices(v, clip_len=32, seg_len=container.streams.video[0].frames)
                    indices = self.sample_frame_indices(video_path=v, clip_len=32, frame_sample_rate=2, seg_len=container.streams.video[0].frames)
                    video = self.read_video_pyav(container=container, indices=indices)
                    video = self.image_processor(video)

                    # Save video to test augs
                    # file_name = v.split('.')[0].split('/')[-1]
                    # frame_rate = 32 #self.video_len/(window_duration)
                    # self.write_video(file_name, frame_rate, video.squeeze())

                    video = video.to(self.device).unsqueeze(0)
                else:
                    video = v.to(self.device)

                videos["video"].append(video)
                videos['attention_mask'].append(1)
            else:
                video = torch.zeros(1, self.video_len, 3, 224, 224).to(self.device)

                videos["video"].append(video)
                videos['attention_mask'].append(0)

        return videos

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs['attention_mask']
        videos = inputs['video']
        videos = torch.cat(videos, 0).float() # type: ignore
        # print(f"videos: {videos.shape}")

        with torch.set_grad_enabled(self.finetune), self.autocast:
            outputs = self.vivit(videos)
            embeds = outputs.last_hidden_state

        empty_idx = torch.LongTensor([i for i, xi in enumerate(mask) if xi == 0])
        mask = torch.ones(embeds.shape[0], embeds.shape[1])
        mask[empty_idx, :] = 0

        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)
        # print(f"\n ViViT mask shape: {mask.shape} \n")
        embeds = (embeds * mask.unsqueeze(-1).to(self.device))
        embeds = self.apply_output_tkns_proj(embeds, name=self.name)

        return embeds, mask

class WaveformConditioner(BaseConditioner):
    """Base class for all conditioners that take a waveform as input.
    Classes that inherit must implement `_get_wav_embedding` that outputs
    a continuous tensor, and `_downsampling_factor` that returns the down-sampling
    factor of the embedding model.

    Args:
        dim (int): The internal representation dimension.
        output_dim (int): Output dimension.
        device (tp.Union[torch.device, str]): Device.
    """
    def __init__(self, dim: int, output_dim: int, device: tp.Union[torch.device, str]):
        super().__init__(dim, output_dim)
        self.device = device
        # if False no masking is done, used in ChromaStemConditioner when completing by periodicity a sample.
        self._use_masking = True

    def tokenize(self, x: WavCondition) -> WavCondition:
        wav, length, sample_rate, path, seek_time = x
        assert length is not None
        return WavCondition(wav.to(self.device), length.to(self.device), sample_rate, path, seek_time)

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        """Gets as input a WavCondition and returns a dense embedding."""
        raise NotImplementedError()

    def _downsampling_factor(self):
        """Returns the downsampling factor of the embedding model."""
        raise NotImplementedError()

    def forward(self, x: WavCondition) -> ConditionType:
        """Extract condition embedding and mask from a waveform and its metadata.
        Args:
            x (WavCondition): Waveform condition containing raw waveform and metadata.
        Returns:
            ConditionType: a dense vector representing the conditioning along with its mask
        """
        wav, lengths, *_ = x
        with torch.no_grad():
            embeds = self._get_wav_embedding(x)
        if hasattr(self, 'output_proj'):
            embeds = embeds.to(self.output_proj.weight)
            embeds = self.output_proj(embeds)

        if lengths is not None and self._use_masking:
            lengths = lengths / self._downsampling_factor()
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore
        else:
            mask = torch.ones_like(embeds[..., 0])
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask

class FeatureExtractor(WaveformConditioner):
    """
    Feature Extractor used for the style conditioner of the paper AUDIO CONDITIONING
        FOR MUSIC GENERATION VIA DISCRETE BOTTLENECK FEATURES.

    Given a waveform, we extract an excerpt of defined length randomly subsampled.
        Then, we feed this excerpt to a feature extractor.

    Args:
        model_name (str): 'encodec' or 'mert'.
        sample_rate (str): sample rate of the input audio. (32000)
        encodec_checkpoint (str): if encodec is used as a feature extractor, checkpoint
            of the model. ('//pretrained/facebook/encodec_32khz' is the default)
        encodec_n_q (int): if encodec is used as a feature extractor it sets the number of
            quantization streams used in it.
        length (float): length in seconds of the random subsampled excerpt that is used
            for conditioning.
        dim (int): The internal representation dimension.
        output_dim (int): Output dimension for the conditioner.
        device (tp.Union[torch.device, str], optional): Device for the conditioner.
        compute_mask (bool): whether to mask the tokens corresponding to the subsampled
            excerpt in the computation of the music language model cross-entropy loss.
        use_middle_of_segment (bool): if True, always take the middle of the input
            instead of a random subsampled excerpt.
        ds_rate_compression (int): downsampling parameter of the compression model used
            for the music language model. (640 for encodec_32khz)
        num_codebooks_lm (int): the number of codebooks used by the music language model.
    """
    def __init__(
        self, model_name: str,
        sample_rate: int, encodec_checkpoint: str, encodec_n_q: int, length: float,
        dim: int, output_dim: int, device: tp.Union[torch.device, str],
        compute_mask: bool = True,
        use_middle_of_segment: bool = False, ds_rate_compression: int = 640,
        num_codebooks_lm: int = 4
    ):
        assert model_name in ['encodec', 'mert']
        if model_name == 'encodec':
            from ..solvers.compression import CompressionSolver
            feat_extractor = CompressionSolver.model_from_checkpoint(encodec_checkpoint, device)
        elif model_name == 'mert':
            from transformers import AutoModel
            feat_extractor = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        super().__init__(
            dim=dim,
            output_dim=output_dim,
            device=device
        )
        self.sample_rate = sample_rate
        self.compute_mask = compute_mask
        self.feat_extractor: nn.Module
        self.embed: tp.Union[nn.ModuleList, nn.Linear]
        if model_name == 'encodec':
            self.__dict__["feat_extractor"] = feat_extractor.to(device)
            self.encodec_n_q = encodec_n_q
            self.embed = nn.ModuleList([nn.Embedding(feat_extractor.cardinality, dim) for _ in range(encodec_n_q)])
        if model_name == 'mert':
            self.__dict__["feat_extractor"] = feat_extractor.eval().to(device)
            self.embed = nn.Linear(768, dim)  # hardcoded
        self.length_subwav = int(length * sample_rate)
        self.ds_rate_compression = ds_rate_compression
        self.model_name = model_name
        self.use_middle_of_segment = use_middle_of_segment
        self.num_codebooks_lm = num_codebooks_lm

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        if x.wav.shape[-1] == 1:
            self.temp_mask = None
            return torch.zeros(x.wav.shape[0], 1, self.dim, device=self.device)
        else:
            with torch.no_grad():
                if self.use_middle_of_segment:
                    start = int((x.wav.shape[-1] - self.length_subwav) / 2)
                    wav = x.wav[:, :, start:start+self.length_subwav]
                else:
                    start = random.randint(0, x.wav.shape[-1] - self.length_subwav)
                    wav = x.wav[:, :, start:start+self.length_subwav]
                if self.compute_mask:
                    self.temp_mask = self._get_mask_wav(x, start)
                if self.model_name == 'encodec':
                    tokens = self.feat_extractor.encode(wav)[0]  # type: ignore
                elif self.model_name == 'mert':
                    wav = convert_audio(wav, from_rate=x.sample_rate[0], to_rate=24000, to_channels=1)
                    embeds = self.feat_extractor(wav.squeeze(-2)).last_hidden_state
            if self.model_name == 'encodec':
                tokens = tokens[:, :self.encodec_n_q]
                embeds = sum([self.embed[k](tokens[:, k]) for k in range(self.encodec_n_q)])  # type: ignore
            else:
                embeds = self.embed(embeds)

            return embeds  # type: ignore # [B, T, dim]

    def _downsampling_factor(self):
        if self.model_name == 'encodec':
            return self.sample_rate / self.feat_extractor.frame_rate
        elif self.model_name == 'mert':
            return self.sample_rate / 75

    def _get_mask_wav(self, x: WavCondition, start: int) -> tp.Union[torch.Tensor, None]:
        if x.wav.shape[-1] == 1:
            return None
        total_length = int(x.wav.shape[-1] / self.ds_rate_compression)
        mask_length = int(self.length_subwav / self.ds_rate_compression)
        start = int(start / self.ds_rate_compression)
        mask = torch.ones(x.wav.shape[0], self.num_codebooks_lm,
                          total_length, device=self.device, dtype=torch.bool)
        mask[:, :, start:start+mask_length] = 0
        return mask

def dropout_condition(sample: ConditioningAttributes,
                      condition_type: str, condition: str,
                      **kwargs) -> ConditioningAttributes:
    """Utility function for nullifying an attribute inside an ConditioningAttributes object.
    If the condition is of type "wav", then nullify it using `nullify_condition` function.
    If the condition is of any other type, set its value to None.
    Works in-place.
    """
    if condition_type not in ['text', 'wav', 'joint_embed', 'video']:
        raise ValueError(
            "dropout_condition got an unexpected condition type!"
            f" expected 'text', 'wav' or 'joint_embed' but got '{condition_type}'"
        )

    if condition not in getattr(sample, condition_type):
        raise ValueError(
            "dropout_condition received an unexpected condition!"
            f" expected wav={sample.wav.keys()} and text={sample.text.keys()}"
            f" but got '{condition}' of type '{condition_type}'!"
        )

    if condition_type == 'wav':
        wav_cond = sample.wav[condition]
        sample.wav[condition] = nullify_wav(wav_cond)
    elif condition_type == 'joint_embed':
        embed = sample.joint_embed[condition]
        sample.joint_embed[condition] = nullify_joint_embed(embed)
    elif condition_type == 'video':
        sample.video[condition] = None
    else:
        sample.text[condition] = None

    return sample


class DropoutModule(nn.Module):
    """Base module for all dropout modules."""
    def __init__(self, seed: int = 1234):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)


class AttributeDropout(DropoutModule):
    """Dropout with a given probability per attribute.
    This is different from the behavior of ClassifierFreeGuidanceDropout as this allows for attributes
    to be dropped out separately. For example, "artist" can be dropped while "genre" remains.
    This is in contrast to ClassifierFreeGuidanceDropout where if "artist" is dropped "genre"
    must also be dropped.

    Args:
        p (tp.Dict[str, float]): A dict mapping between attributes and dropout probability. For example:
            ...
            "genre": 0.1,
            "artist": 0.5,
            "wav": 0.25,
            ...
        active_on_eval (bool, optional): Whether the dropout is active at eval. Default to False.
        seed (int, optional): Random seed.
    """
    def __init__(self, p: tp.Dict[str, tp.Dict[str, float]], active_on_eval: bool = False, seed: int = 1234):
        super().__init__(seed=seed)
        self.active_on_eval = active_on_eval
        # construct dict that return the values from p otherwise 0
        self.p = {}
        for condition_type, probs in p.items():
            self.p[condition_type] = defaultdict(lambda: 0, probs)

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after certain attributes were set to None.
        """
        if not self.training and not self.active_on_eval:
            return samples

        samples = deepcopy(samples)
        for condition_type, ps in self.p.items():  # for condition types [text, wav, symbolic]
            for condition, p in ps.items():  # for attributes of each type (e.g., [artist, genre])
                if torch.rand(1, generator=self.rng).item() < p:
                    for sample in samples:
                        dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"AttributeDropout({dict(self.p)})"


class ClassifierFreeGuidanceDropout(DropoutModule):
    """Classifier Free Guidance dropout.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """
    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p

    def forward(self, samples: tp.List[ConditioningAttributes],
                cond_types: tp.List[str] = ["wav", "text", "video"],
                **kwargs) -> tp.List[ConditioningAttributes]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after all attributes were set to None.
        """
        if not self.training:
            return samples

        # decide on which attributes to drop in a batched fashion
        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return samples

        # nullify conditions of all attributes
        samples = deepcopy(samples)
        for condition_type in cond_types:
            for sample in samples:
                for condition in sample.attributes[condition_type]:
                    dropout_condition(sample, condition_type, condition, **kwargs)
        return samples

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"


class ConditioningProvider(nn.Module):
    """Prepare and provide conditions given all the supported conditioners.

    Args:
        conditioners (dict): Dictionary of conditioners.
        device (torch.device or str, optional): Device for conditioners and output condition types.
    """
    def __init__(
            self, 
            conditioners: tp.Dict[str, BaseConditioner], 
            device: tp.Union[torch.device, str] = "cpu", 
        ):
        super().__init__()
        self.device = device
        self.conditioners = nn.ModuleDict(conditioners)

    @property
    def joint_embed_conditions(self):
        return [m.attribute for m in self.conditioners.values() if isinstance(m, JointEmbeddingConditioner)]

    @property
    def has_joint_embed_conditions(self):
        return len(self.joint_embed_conditions) > 0

    @property
    def text_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, TextConditioner)]

    @property
    def video_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, VideoConditioner)]

    @property
    def wav_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, WaveformConditioner)]

    @property
    def has_wav_condition(self):
        return len(self.wav_conditions) > 0

    def tokenize(self, inputs: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.Any]:
        """Match attributes/wavs with existing conditioners in self, and compute tokenize them accordingly.
        This should be called before starting any real GPU work to avoid synchronization points.
        This will return a dict matching conditioner names to their arbitrary tokenized representations.

        Args:
            inputs (list[ConditioningAttributes]): List of ConditioningAttributes objects containing
                text and wav conditions.
        """
        assert all([isinstance(x, ConditioningAttributes) for x in inputs]), (
            "Got unexpected types input for conditioner! should be tp.List[ConditioningAttributes]",
            f" but types were {set([type(x) for x in inputs])}"
        )

        output = {}
        text = self._collate_text(inputs)
        video = self._collate_video(inputs)
        wavs = self._collate_wavs(inputs)
        joint_embeds = self._collate_joint_embeds(inputs)

        assert set(text.keys() | video.keys() | wavs.keys() | joint_embeds.keys()).issubset(set(self.conditioners.keys())), (
            f"Got an unexpected attribute! Expected {self.conditioners.keys()}, ",
            f"got {text.keys(), wavs.keys(), joint_embeds.keys()}"
        )

        for attribute, batch in chain(text.items(), video.items(), wavs.items(), joint_embeds.items()):
            output[attribute] = self.conditioners[attribute].tokenize(batch)
        return output

    def forward(self, tokenized: tp.Dict[str, tp.Any]) -> tp.Dict[str, ConditionType]:
        """Compute pairs of `(embedding, mask)` using the configured conditioners and the tokenized representations.
        The output is for example:
        {
            "genre": (torch.Tensor([B, 1, D_genre]), torch.Tensor([B, 1])),
            "description": (torch.Tensor([B, T_desc, D_desc]), torch.Tensor([B, T_desc])),
            ...
        }

        Args:
            tokenized (dict): Dict of tokenized representations as returned by `tokenize()`.
        """
        output = {}
        for attribute, inputs in tokenized.items():
            condition, mask = self.conditioners[attribute](inputs)
            output[attribute] = (condition, mask)
        return output

    def _collate_text(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.List[tp.Optional[str]]]:
        """Given a list of ConditioningAttributes objects, compile a dictionary where the keys
        are the attributes and the values are the aggregated input per attribute.
        For example:
        Input:
        [
            ConditioningAttributes(text={"genre": "Rock", "description": "A rock song with a guitar solo"}, wav=...),
            ConditioningAttributes(text={"genre": "Hip-hop", "description": "A hip-hop verse"}, wav=...),
        ]
        Output:
        {
            "genre": ["Rock", "Hip-hop"],
            "description": ["A rock song with a guitar solo", "A hip-hop verse"]
        }

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, list[str, optional]]: A dictionary mapping an attribute name to text batch.
        """
        out: tp.Dict[str, tp.List[tp.Optional[str]]] = defaultdict(list)
        texts = [x.text for x in samples]
        for text in texts:
            for condition in self.text_conditions:
                out[condition].append(text[condition])
        return out

    def _collate_video(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.List[tp.Optional[str]]]:
        """Given a list of ConditioningAttributes objects, compile a dictionary where the keys
        are the attributes and the values are the aggregated input per attribute.
        For example:
        Input:
        [
            ConditioningAttributes(video={"visual_content": "/data1/1.mp4"}, wav=...),
            ConditioningAttributes(video={"visual_content": "/data1/2.mp4"}, wav=...),
        ]
        Output:
        {
            "visual_content": ["/data1/1.mp4", "/data1/2.mp4"]
        }

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, list[str, optional]]: A dictionary mapping an attribute name to text batch.
        """
        out: tp.Dict[str, tp.List[tp.Optional[str]]] = defaultdict(list)

        videos = [x.video for x in samples]

        for video in videos:
            for condition in self.video_conditions:
                out[condition].append(video[condition])

        return out

    def _collate_wavs(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, WavCondition]:
        """Generate a dict where the keys are attributes by which we fetch similar wavs,
        and the values are Tensors of wavs according to said attributes.

        *Note*: by the time the samples reach this function, each sample should have some waveform
        inside the "wav" attribute. It should be either:
        1. A real waveform
        2. A null waveform due to the sample having no similar waveforms (nullified by the dataset)
        3. A null waveform due to it being dropped in a dropout module (nullified by dropout)

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, WavCondition]: A dictionary mapping an attribute name to wavs.
        """
        wavs = defaultdict(list)
        lengths = defaultdict(list)
        sample_rates = defaultdict(list)
        paths = defaultdict(list)
        seek_times = defaultdict(list)
        out: tp.Dict[str, WavCondition] = {}

        for sample in samples:
            for attribute in self.wav_conditions:
                wav, length, sample_rate, path, seek_time = sample.wav[attribute]
                assert wav.dim() == 3, f"Got wav with dim={wav.dim()}, but expected 3 [1, C, T]"
                assert wav.size(0) == 1, f"Got wav [B, C, T] with shape={wav.shape}, but expected B == 1"
                # mono-channel conditioning
                wav = wav.mean(1, keepdim=True)  # [1, 1, T]
                wavs[attribute].append(wav.flatten())  # [T]
                lengths[attribute].append(length)
                sample_rates[attribute].extend(sample_rate)
                paths[attribute].extend(path)
                seek_times[attribute].extend(seek_time)

        # stack all wavs to a single tensor
        for attribute in self.wav_conditions:
            stacked_wav, _ = collate(wavs[attribute], dim=0)
            out[attribute] = WavCondition(
                stacked_wav.unsqueeze(1), torch.cat(lengths[attribute]), sample_rates[attribute],
                paths[attribute], seek_times[attribute])

        return out

    def _collate_joint_embeds(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, JointEmbedCondition]:
        """Generate a dict where the keys are attributes by which we compute joint embeddings,
        and the values are Tensors of pre-computed embeddings and the corresponding text attributes.

        Args:
            samples (list[ConditioningAttributes]): List of ConditioningAttributes samples.
        Returns:
            A dictionary mapping an attribute name to joint embeddings.
        """
        texts = defaultdict(list)
        video = defaultdict(list)
        wavs = defaultdict(list)
        lengths = defaultdict(list)
        sample_rates = defaultdict(list)
        paths = defaultdict(list)
        seek_times = defaultdict(list)
        channels: int = 0

        out = {}
        for sample in samples:
            for attribute in self.joint_embed_conditions:
                wav, text, video, length, sample_rate, path, seek_time = sample.joint_embed[attribute]
                assert wav.dim() == 3
                if channels == 0:
                    channels = wav.size(1)
                else:
                    assert channels == wav.size(1), "not all audio has same number of channels in batch"
                assert wav.size(0) == 1, "Expecting single-wav batch in the collate method"
                wav = einops.rearrange(wav, "b c t -> (b c t)")  # [1, C, T] => [C * T]
                wavs[attribute].append(wav)
                texts[attribute].extend(text)
                video[attribute].extend(video)
                lengths[attribute].append(length)
                sample_rates[attribute].extend(sample_rate)
                paths[attribute].extend(path)
                seek_times[attribute].extend(seek_time)

        for attribute in self.joint_embed_conditions:
            stacked_texts = texts[attribute]
            stacked_video = video[attribute]
            stacked_paths = paths[attribute]
            stacked_seek_times = seek_times[attribute]
            stacked_wavs = pad_sequence(wavs[attribute]).to(self.device)
            stacked_wavs = einops.rearrange(stacked_wavs, "(c t) b -> b c t", c=channels)
            stacked_sample_rates = sample_rates[attribute]
            stacked_lengths = torch.cat(lengths[attribute]).to(self.device)
            assert stacked_lengths.size(0) == stacked_wavs.size(0)
            assert len(stacked_sample_rates) == stacked_wavs.size(0)
            assert len(stacked_texts) == stacked_wavs.size(0)
            out[attribute] = JointEmbedCondition(
                text=stacked_texts, 
                video=stacked_video, 
                wav=stacked_wavs,
                length=stacked_lengths, 
                sample_rate=stacked_sample_rates,
                path=stacked_paths, 
                seek_time=stacked_seek_times
            )

        return out


class ConditionFuser(StreamingModule):
    """Condition fuser handles the logic to combine the different conditions
    to the actual model input.

    Args:
        fuse2cond (tp.Dict[str, str]): A dictionary that says how to fuse
            each condition. For example:
            {
                "prepend": ["description"],
                "sum": ["genre", "bpm"],
                "cross": ["description"],
            }
        cross_attention_pos_emb (bool, optional): Use positional embeddings in cross attention.
        cross_attention_pos_emb_scale (int): Scale for positional embeddings in cross attention if used.
    """
    FUSING_METHODS = ["sum", "prepend", "cross", "cross_sum", "ignore", "input_interpolate"]

    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]], cross_attention_pos_emb: bool = False,
                 cross_attention_pos_emb_scale: float = 1.0):
        super().__init__()
        assert all(
            [k in self.FUSING_METHODS for k in fuse2cond.keys()]
        ), f"Got invalid fuse method, allowed methods: {self.FUSING_METHODS}"
        self.cross_attention_pos_emb = cross_attention_pos_emb
        self.cross_attention_pos_emb_scale = cross_attention_pos_emb_scale
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method

    def forward(
        self,
        input: torch.Tensor,
        conditions: tp.Dict[str, ConditionType]
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Fuse the conditions to the provided model input.

        Args:
            input (torch.Tensor): Transformer input.
            conditions (dict[str, ConditionType]): Dict of conditions.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The first tensor is the transformer input
                after the conditions have been fused. The second output tensor is the tensor
                used for cross-attention or None if no cross attention inputs exist.
        """
        B, T, _ = input.shape

        if 'offsets' in self._streaming_state:
            first_step = False
            offsets = self._streaming_state['offsets']
        else:
            first_step = True
            offsets = torch.zeros(input.shape[0], dtype=torch.long, device=input.device)

        assert set(conditions.keys()).issubset(set(self.cond2fuse.keys())), \
            f"given conditions contain unknown attributes for fuser, " \
            f"expected {self.cond2fuse.keys()}, got {conditions.keys()}"

        cross_attention_output = None
        for cond_type, (cond, cond_mask) in conditions.items():
            op = self.cond2fuse[cond_type]            
            if op == 'sum':
                input += cond
            elif op == 'input_interpolate':
                cond = einops.rearrange(cond, "b t d -> b d t")
                cond = F.interpolate(cond, size=input.shape[1])
                input += einops.rearrange(cond, "b d t -> b t d")
            elif op == 'prepend':
                if first_step:
                    input = torch.cat([cond, input], dim=1)
            elif op == 'cross':
                if cross_attention_output is not None:
                    cross_attention_output = torch.cat([cross_attention_output, cond], dim=1)
                else:
                    cross_attention_output = cond
            elif op == 'cross_sum':
                if cross_attention_output is not None:
                    cross_attention_output += cond
                else:
                    cross_attention_output = cond
            elif op == 'ignore':
                continue
            else:
                raise ValueError(f"unknown op ({op})")

        if self.cross_attention_pos_emb and cross_attention_output is not None:
            positions = torch.arange(
                cross_attention_output.shape[1],
                device=cross_attention_output.device
            ).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, cross_attention_output.shape[-1])
            cross_attention_output = cross_attention_output + self.cross_attention_pos_emb_scale * pos_emb

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return input, cross_attention_output
