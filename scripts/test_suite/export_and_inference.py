import os

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.utils import export
from audiocraft import train

def export_musicgen_tune(sig:str, base_save_path:str) -> str:
    """
        To export XP as .bin files for inference after a training run
        from https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md#importing--exporting-models

        Args:
            sig: xp sig
            base_save_path: Path to save the extracted checkpoint

        Returns:
            checkpoint path from where one can load the finetuned exported model for inference
    """
    xp = train.main.get_xp_from_sig(sig)
    if sig.startswith('facebook'):
        xp_path = xp.folder
    else:
        xp_path = '/'.join(str(xp.folder).split('/')[:-1] + [sig, 'checkpoint.th'])
    print(f"XP PATH {xp_path}")

    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    lm_path = os.path.join(base_save_path, 'state_dict.bin')
    if not os.path.isfile(lm_path):
        export.export_lm(xp_path, lm_path)

    compress_path = os.path.join(base_save_path, 'compression_state_dict.bin')
    if not os.path.isfile(compress_path):
        export.export_pretrained_compression_model('facebook/encodec_32khz', compress_path)

    return base_save_path

def run_inference_musicgen(sig:str, desc, save_path, checkpoint_path):
    model_path = sig

    if not sig.startswith("facebook/"):
        model_path = export_musicgen_tune(sig, checkpoint_path)

    musicgen = MusicGen.get_pretrained(model_path)
    musicgen.set_generation_params(
        duration=11,
    )

    wave = musicgen.generate([desc])[0]

    audio_write(save_path, wave.cpu(), musicgen.sample_rate, strategy="loudness", loudness_compressor=True)