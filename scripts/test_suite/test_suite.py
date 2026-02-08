import os
import json
import argparse
import random
from datetime import datetime
import shutil

import pandas as pd
import numpy as np
from tqdm import tqdm

from export_and_inference import run_inference_musicgen

import moviepy as mp
from pydub import AudioSegment

def is_mp3(file:str):
    extension = file.split('.')[-1]
    return extension == 'mp3'

def get_genre(genres_df, game):
    genre = genres_df[genres_df['game_folder'] == game]
    genre = genre['game_genre'].to_numpy()
    genre = np.random.choice(genre, 1)[0]

    return genre

def read_dataset_split(dataset_split_path:str, genres_path:str) -> list[dict[str, str]]:
    """
        Returns:
            list of dicts containing relevant information about the samples, like the video path, the audio path and the description
    """
    dataset_split_path = os.path.abspath(dataset_split_path)
    samples_dicts:list[dict[str, str]] = []
    genres_df = pd.read_csv(genres_path)

    for file in sorted(os.listdir(dataset_split_path)):
        if is_mp3(file):
            continue

        json_path = os.path.join(dataset_split_path, file)
        json_dict:dict[str, str] = {}
        with open(json_path, 'r') as f:
            general_json_dict = json.load(f)

            json_dict['game'] = general_json_dict['name'].split('_')[0]
            json_dict['genre'] = get_genre(genres_df, json_dict['game'])
            json_dict['video'] = general_json_dict['video']
            json_dict['audio'] = os.path.join(dataset_split_path, general_json_dict['name'])
            json_dict['description'] = general_json_dict['description']

        samples_dicts.append(json_dict)

    return samples_dicts

def get_one_sample_per_game(samples_dicts:list[dict[str, str]]) -> list[dict[str, str]]:
    # dumb dict in order to process the last game in the samples_dicts list
    none_dict = {'game': "NONE"}
    samples_dicts.append(none_dict)

    choosen_samples:list[dict[str, str]] = []

    current_game = ''
    game_dicts = []

    for sample_dict in samples_dicts:
        game = sample_dict['game']

        if current_game == '':
            current_game = game

        if game == current_game:
            game_dicts.append(sample_dict)
        else:
            choosen_sample = random.choice(game_dicts)
            choosen_samples.append(choosen_sample)
            game_dicts.clear()

            current_game = game
            game_dicts.append(sample_dict)

    return choosen_samples

def run_inference(samples_dicts:list[dict[str, str]], model_sig, base_path:str):
    # Save audios folder structure
    # model_date
    #   |_genre
    #     |_game
    #         |_vid_cp.mp4
    #         |_orig_sdtk.mp3
    #         |_gen_sdtk.wav
    #         |_metadata.json

    inference_path = os.path.join(base_path, 'inference')
    checkpoint_path = os.path.join(base_path, 'checkpoint')

    for sample_dict in tqdm(samples_dicts):
        vid_name = sample_dict['video'].split('/')[-1][:-4]
        vid_folder_path = os.path.join(inference_path, sample_dict['genre'], sample_dict['game'])

        if not os.path.exists(vid_folder_path):
            os.makedirs(vid_folder_path)

        print(vid_folder_path)

        # Copy video
        vid_path = vid_folder_path + f'/{vid_name}.mp4'
        if not os.path.exists(vid_path):
            shutil.copy(sample_dict['video'], vid_path)

        # Copy soundtrack
        orig_sdtk = vid_folder_path + f'/{vid_name}.mp3'
        if not os.path.exists(orig_sdtk):
            shutil.copy(sample_dict['audio'], orig_sdtk)

        # Create description txt
        desc_path = vid_folder_path + f'/{vid_name}.txt'
        desc = sample_dict['description']
        if not os.path.exists(desc_path):
            with open(desc_path, 'w') as f:
                f.write(desc)

        vid = sample_dict['video']

        # Generate sountrack
        gen_sdtk = vid_folder_path + f'/{vid_name}_gen'
        if not os.path.exists(gen_sdtk):
            run_inference_musicgen(model_sig, desc, vid, gen_sdtk, checkpoint_path)

        video_mp = mp.VideoFileClip(vid_path)
        audio_clip = AudioSegment.from_wav(gen_sdtk+'.wav')
        audio_clip[0:int(video_mp.duration*1000)].export(gen_sdtk+'.wav')
        # Render generated music into input video
        audio_mp = mp.AudioFileClip(gen_sdtk+'.wav')

        audio_mp = audio_mp.subclipped(0, video_mp.duration )
        final = video_mp.without_audio()
        final = video_mp.with_audio(audio_mp)
        try:
            final.write_videofile(os.path.join(vid_folder_path, vid_name+'_gen.mp4'),
                codec='libx264', 
                audio_codec='aac', 
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
        except Exception as e:
            print(f"error：{e}")
        #os.remove(gen_sdtk+'.wav')

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="/home/es119256/dados/xps/checkpoints_and_inference", help="path to folder where results will be stored")
    parser.add_argument('--genres_path', type=str, default="/home/es119256/dados/datasets/vmdb_3/deepseek_genres.csv", help="path to games genres csv")
    parser.add_argument('--split', type=str, default="test", help="split to be accessed in dataset/snes_mvdb/SPLIT")
    parser.add_argument('--converted_dataset', type=str, default="/home/es119256/dados/repos/visual-bardo-video/dataset", help="path to audiocraft/dataset. snes_mvdb will be added to access the converted dataset")
    parser.add_argument('--model_sig', type=str, default="t5_vivit_musicgen_tuned_wO_t5_865e739b", help="path to checkpoint")
    parser.add_argument('--is_vanilla', type=bool, default=False, help="If True model_checkpoint will be facebook/musicgen-medium")

    args = parser.parse_args()

    genres_path = args.genres_path
    dataset_split_path = os.path.join(args.converted_dataset, 'snes_mvdb', args.split)

    model_sig = 'facebook/musicgen-medium'
    model_name = "vanilla_medium"
    if args.is_vanilla == False:
        model_sig = args.model_sig
        model_name = args.model_sig

    date = datetime.now()
    date = date.strftime("%m_%d_%y")
    base_path_path = os.path.join(args.base_path, f'{model_name}_{date}')

    print(f"dataset_split_path: {dataset_split_path}")
    print(f"model_sig: {model_sig}")

    random.seed(42)

    # Read dataset split, select samples and run inference
    samples_dicts = read_dataset_split(dataset_split_path, genres_path)
    samples_dicts = get_one_sample_per_game(samples_dicts)

    # for sample_dict in samples_dicts:
    #     print(sample_dict['game'], sample_dict['audio'], sample_dict['genre'])

    run_inference(samples_dicts, model_sig, base_path_path)

if __name__ == "__main__":
    main()