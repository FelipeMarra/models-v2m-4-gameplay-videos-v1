import subprocess
import sys

models = [
    #("/home/es119256/dados/xps/audiocraft_felipe/xps/t5_musicgen_retrain_d6698d5d", "T5->MusicGen_Retrain"),
    #("/home/es119256/dados/xps/audiocraft_felipe/xps/t5_musicgen_retrain_wO_t5_5aa5e26a", "T5->MusicGen_wO_T5_Retrain"),
    #("/home/es119256/dados/xps/audiocraft_felipe/xps/t5_musicgen_tuned_5e224a89", "T5->MusicGen_Tuned"),
    #("/home/es119256/dados/xps/audiocraft_felipe/xps/t5_musicgen_tuned_wO_t5_3f93db94", "T5->MusicGen_wO_T5_Tuned"),

    #("/home/es119256/dados/xps/audiocraft_frozen_felipe/xps/5aa5e26a_t5_adaptor_random", "T5+Adapter->MusicGen_Retrain"),
    ("/home/es119256/dados/xps/audiocraft_frozen_felipe/xps/5e224a89_t5_adaptor", "T5+Adapter->MusicGen_Tuned"),

    #("/home/es119256/dados/xps/audiocraft_vivit_felipe/xps/vivit_musicgen_retrain_fc1b8bab", "ViViT->MusicGen_Retrain"),
    #("/home/es119256/dados/xps/audiocraft_vivit_felipe/xps/vivit_musicgen_tuned_e74c0c91", "ViViT->MusicGen_Tuned"),

    #("/home/es119256/dados/xps/audiocraft_vivit_t5_felipe/xps/t5_vivit_musicgen_tuned_wO_t5_78bf6732", "ViViT+T5->MusicGen_wO_T5_Tuned"),

    ("/home/es119256/dados/xps/audiocraft_felipe/xps/60c93052", "T5->MusicGen_Tuned_TRAD_EPOCH"),
    ("/home/es119256/dados/xps/audiocraft_vivit_t5_felipe/xps/dea6409f", "ViViT+T5->MusicGen_Retrain"),
]

script = '/home/es119256/dados/repos/visual-bardo-video/scripts/test_suite/test_suite.py'

for model in models:
    subprocess.run([sys.executable, script, "--model_path", model[0], "--model_name", model[1]])

# subprocess.run([sys.executable, script, "--is_vanilla", str(True)])