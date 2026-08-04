# Video to Music Generation for Gameplay Videos
## Introduction
This is the code repository for the paper Video to Music Generation for Gameplay Videos.

> Paper: #TODO

> Demo: [demo-v2m-4-gameplay-videos-v1](https://github.com/FelipeMarra/demo-v2m-4-gameplay-videos-v1)

> Dataset: [dataset-v2m-4-gameplay-videos-v1](https://github.com/FelipeMarra/dataset-v2m-4-gameplay-videos-v1)

We present three different approaches for autoregressive music generation for gameplay videos, namely: Text as Interface, Text Embedding as Interface and Direct Mapping. This repository contains the code for the following models: 🔥T5, ❄️T5, 🔥ViT, ❄️ViT, 🔥ViViT and ❄️ViViT - nomeclature as presented in the paper. This repo is a fork of Meta's [Audiocraft](https://github.com/facebookresearch/audiocraft), as thsese models are bult on top of the MusicGen model. We vary the encoders as indicated by the models' names. The 🔥 simble indicates that the encoder was trained, while the ❄️ simble indicates that the encoder remained frozen during training. The decoder is the same as in the MusicGen model and it is always trained.

## General Repository Structure
To generate the dataset and get the jsons in the format expected by [Audiocraft](https://github.com/facebookresearch/audiocraft), head towards our [dataset repo](https://github.com/FelipeMarra/dataset-v2m-4-gameplay-videos-v1).

The commands used to train and evaluate the models can befound in the [cmd](/cmd) folder.

The different encoders can be found in the [conditioners](/audiocraft/modules/conditioners.py) file.

## Main Models' files
### 🔥 and ❄️ T5
#TODO

### 🔥 and ❄️ ViT
#TODO

### 🔥 and ❄️ ViViT
#TODO

## Workarounds
### Multiple Examples for the Same Audio
We have multiple videos mapped to the same audio, therefore we need to load all the jsons from `dataset/snes_mvdb/split`. For that we had to directly change a file in the laion_clab lib in the site-packages.

* Add the [json_path property to AudioMeta](/audiocraft/data/audio_dataset.py#L61)

* Add None to when [returning the AudioMeta in he _get_audio_meta function](/audiocraft/data/audio_dataset.py#L120)

* Change `music_info_path` in `MusicDataset` [to get the `json_path` property](/audiocraft/data/music_dataset.py#L231).

At /root/miniconda3/lib/python3.9/site-packages/laion_clap/clap_module/factory.py, replace at line 63 by:
```python
        # removing position_ids to maintain compatibility with latest transformers update        
        if version.parse(transformers.__version__) >= version.parse("4.31.0"):
            if state_dict.get("text_branch.embeddings.position_ids") != None:
                del state_dict["text_branch.embeddings.position_ids"]
```

### Adaptation for Video
#TODO