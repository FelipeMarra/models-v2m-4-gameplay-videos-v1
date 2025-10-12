# Visual Bardo
Text-as-Interface Visual Bardo.
Fine tuning [MusicGen along with T5](https://github.com/facebookresearch/audiocraft) in the [SNES MVDB dataset](https://github.com/jknvlvxs/vmdb).

# Modification
One should head towards Audiocraft's repo for the original documentation. The following only lists the adaptations in their code to work with the SNES MVDB dataset.

## Multiple Examples for the Same Audio
We have multiple videos mapped to the same audio, therefore we need to load all the jsons from `dataset/snes_mvdb/split`.

* Add the [json_path property to AudioMeta](https://github.com/FelipeMarra/visual-bardo/blob/ddcbaaac6275d205a5b242c0d72a378e89a67308/audiocraft/data/audio_dataset.py#L61)

* Add None to when [returning the AudioMeta in he _get_audio_meta function](https://github.com/FelipeMarra/visual-bardo/blob/ddcbaaac6275d205a5b242c0d72a378e89a67308/audiocraft/data/audio_dataset.py#L120)

* Change `music_info_path` in `MusicDataset` [to get the `json_path` property](https://github.com/FelipeMarra/visual-bardo/blob/ddcbaaac6275d205a5b242c0d72a378e89a67308/audiocraft/data/music_dataset.py#L231).

At /root/miniconda3/lib/python3.9/site-packages/laion_clap/clap_module/factory.py, replace at line 63 by:
```python
        # removing position_ids to maintain compatibility with latest transformers update        
        if version.parse(transformers.__version__) >= version.parse("4.31.0"):
            if state_dict.get("text_branch.embeddings.position_ids") != None:
                del state_dict["text_branch.embeddings.position_ids"]
```