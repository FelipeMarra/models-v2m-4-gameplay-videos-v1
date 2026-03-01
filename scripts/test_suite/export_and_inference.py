import os

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.utils import export

def export_musicgen_tune(model_path:str, save_checkpoint_path:str) -> str:
    """
        To export XP as .bin files for inference after a training run
        from https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md#importing--exporting-models

        Args:
            model_path: path to model folder
            save_checkpoint_path: Path to save the extracted checkpoint

        Returns:
            checkpoint path from where one can load the finetuned exported model for inference
    """
    if not os.path.exists(save_checkpoint_path):
        os.makedirs(save_checkpoint_path)

    lm_path = os.path.join(save_checkpoint_path, 'state_dict.bin')
    if not os.path.isfile(lm_path):
        export.export_lm(model_path+'/checkpoint.th', lm_path)

    compress_path = os.path.join(save_checkpoint_path, 'compression_state_dict.bin')
    if not os.path.isfile(compress_path):
        export.export_pretrained_compression_model('facebook/encodec_32khz', compress_path)

    return save_checkpoint_path

def run_inference_musicgen(model_path:str, desc, vid, save_path, save_checkpoint_path):
    if not model_path.startswith("facebook/"):
        model_path = export_musicgen_tune(model_path, save_checkpoint_path)

    musicgen = MusicGen.get_pretrained(model_path)
    musicgen.set_generation_params(
        duration=11,
    )

    wave = musicgen.generate([desc], [vid])[0]

    audio_write(save_path, wave.cpu(), musicgen.sample_rate, strategy="loudness", loudness_compressor=True)