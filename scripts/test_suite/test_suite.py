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

import moviepy.editor as mp
from pydub import AudioSegment

def is_mp3(file:str):
    extension = file.split('.')[-1]
    return extension == 'mp3'

def get_genre(genres_df, game):
    genre = genres_df[genres_df['game_folder'] == game]
    genre = genre['game_genre'].to_numpy()
    genre = np.random.choice(genre, 1)[0]

    return genre

def read_dataset_split(dataset_split_path:str, genres_path:str) -> dict[str, dict[str, dict[str, list[dict]]]]:
    """
        Returns:
            list of dicts containing relevant information about the samples, like the video path, the audio path and the description
    """
    dataset_split_path = os.path.abspath(dataset_split_path)
    # samples_dicts structure
    # {
    #     genre_1: {
    #         game_1: {
    #             audio_1: [game_1_audio_1_content_1, game_1_audio_1_content_2...]
    #         }
    #     }
    # }
    samples_dicts:dict[str, dict[str, dict[str, list[dict]]]] = {}
    genres_df = pd.read_csv(genres_path)

    for file in sorted(os.listdir(dataset_split_path)):
        if is_mp3(file):
            continue

        json_path = os.path.join(dataset_split_path, file)
        game_content:dict[str, str] = {}
        with open(json_path, 'r') as f:
            general_json_dict = json.load(f)

            game = general_json_dict['name'].split('_')[0]
            genre = get_genre(genres_df, game)
            audio = os.path.join(dataset_split_path, general_json_dict['name'])

            game_content['video'] = general_json_dict['video']
            game_content['description'] = general_json_dict['description']

        if not samples_dicts.get(genre):
            samples_dicts[genre] = {}
        if not samples_dicts[genre].get(game):
            samples_dicts[genre][game] = {}
        if not samples_dicts[genre][game].get(audio):
            samples_dicts[genre][game][audio] = []


        samples_dicts[genre][game][audio].append(game_content)

    return samples_dicts

def get_n_samples_per_game(samples_dicts:dict[str, dict[str, dict[str, list[dict]]]], n:int) -> list[dict[str, str]]:
    choosen_samples:list[dict[str, str]] = []

    for genre_name, genre_games in samples_dicts.items():
        for game_name, game_audios in genre_games.items():
            choosen_audios = game_audios

            # Make sure to get at most 3 different soundtracks
            if len(game_audios.keys()) > n:
                choosen_audios = {}
                chosen_audios_keys = random.choices(list(game_audios.keys()), k=n)
                for key in chosen_audios_keys:
                    choosen_audios[key] = game_audios[key]

            for audio_name, audio_content in choosen_audios.items():
                # Get random video for current soundtrack
                choosen_audio_content = random.choice(audio_content)

                choosen_sample = {
                    'game': game_name,
                    'genre': genre_name,
                    'audio': audio_name,
                    'video': choosen_audio_content['video'],
                    'description': choosen_audio_content['description']
                }

                choosen_samples.append(choosen_sample)

    return choosen_samples

def run_inference(samples_dicts:list[dict[str, str]], model_path, base_path:str, gt_base_path:str, get_gt:bool):
    # Save audios folder structure
    # model_date
    #   |_genre
    #     |_game
    #         |_vid_cp.mp4
    #         |_orig_sdtk.mp3
    #         |_gen_sdtk.wav
    #         |_metadata.json

    inference_path = os.path.join(base_path, 'inference')
    save_checkpoint_path = os.path.join(base_path, 'checkpoint')

    for sample_dict in tqdm(samples_dicts):
        vid_name = sample_dict['video'].split('/')[-1][:-4]
        desc = sample_dict['description']
        vid_gt_folder_path = os.path.join(gt_base_path, 'inference', sample_dict['genre'], sample_dict['game'])
        vid_path = vid_gt_folder_path + f'/{vid_name}.mp4'

        if get_gt:
            if not os.path.exists(vid_gt_folder_path):
                os.makedirs(vid_gt_folder_path)

            print('Copying GT Vid:', vid_gt_folder_path)

            # Copy video
            if not os.path.exists(vid_path):
                shutil.copy(sample_dict['video'], vid_path)

            # Copy soundtrack
            orig_sdtk = vid_gt_folder_path + f'/{vid_name}.mp3'
            if not os.path.exists(orig_sdtk):
                shutil.copy(sample_dict['audio'], orig_sdtk)

            # Create description txt
            desc_path = vid_gt_folder_path + f'/{vid_name}.txt'
            if not os.path.exists(desc_path):
                with open(desc_path, 'w') as f:
                    f.write(desc)

        # Generate sdtk and video for model
        vid_folder_path = os.path.join(inference_path, sample_dict['genre'], sample_dict['game'])

        if not os.path.exists(vid_folder_path):
            os.makedirs(vid_folder_path)

        print('Generating Vid For:', vid_folder_path)

        vid = sample_dict['video']

        # Generate sountrack
        gen_sdtk = vid_folder_path + f'/{vid_name}_gen'
        if not os.path.exists(gen_sdtk):
            run_inference_musicgen(model_path, desc, vid, gen_sdtk, save_checkpoint_path)

        video_mp = mp.VideoFileClip(vid_path)
        audio_clip = AudioSegment.from_wav(gen_sdtk+'.wav')
        audio_clip[0:int(video_mp.duration*1000)].export(gen_sdtk+'.wav')
        # Render generated music into input video
        audio_mp = mp.AudioFileClip(gen_sdtk+'.wav')

        audio_mp = audio_mp.subclip(0, video_mp.duration )
        final = video_mp.without_audio()
        final = video_mp.set_audio(audio_mp)
        try:
            final.write_videofile(os.path.join(vid_folder_path, vid_name+'_gen.mp4'),
                codec='libx264', 
                audio_codec='aac', 
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
        except Exception as e:
            print(f"error：{e}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="/home/es119256/dados/xps/checkpoints_and_inference_final", help="path to folder where results will be stored")
    parser.add_argument('--genres_path', type=str, default="/home/es119256/dados/datasets/vmdb_3/deepseek_genres.csv", help="path to games genres csv")
    parser.add_argument('--split', type=str, default="test", help="split to be accessed in dataset/snes_mvdb/SPLIT")
    parser.add_argument('--converted_dataset', type=str, default="/home/es119256/dados/repos/visual-bardo-video/dataset", help="path to audiocraft/dataset. snes_mvdb will be added to access the converted dataset")
    parser.add_argument('--is_vanilla', type=bool, default=False, help="If True model_checkpoint will be facebook/musicgen-medium")
    parser.add_argument('--model_path', type=str, help="path to checkpoint")
    parser.add_argument('--model_name', type=str, help="model folder name inside base_path")

    args = parser.parse_args()

    genres_path = args.genres_path
    dataset_split_path = os.path.join(args.converted_dataset, 'snes_mvdb', args.split)

    model_path = 'facebook/musicgen-medium'
    model_name = "T5->MusicGen_Base"
    if args.is_vanilla == False:
        model_path = args.model_path
        model_name = args.model_name

    date = datetime.now()
    date = date.strftime("%m_%d_%y")
    base_path = os.path.join(args.base_path, f'{model_name}_{date}')

    gt_base_path = os.path.join(args.base_path, 'Ground_Truth')
    if os.path.exists(gt_base_path):
        get_gt = False
    else:
        get_gt = True
        print(f"craeting gt: {gt_base_path}")

    print(f"dataset_split_path: {dataset_split_path}")
    print(f"model_path: {model_path} | model_name: {model_name}")

    random.seed(42)

    # Read dataset split, select samples and run inference
    samples_dicts = read_dataset_split(dataset_split_path, genres_path)
    samples_dicts = get_n_samples_per_game(samples_dicts, 3)

    # # Debug
    # game = ''
    # n_games = 0
    # n_samples = 0
    # for sample_dict in samples_dicts:
    #     append = ''
    #     if sample_dict['game'] != game: 
    #         game = sample_dict['game']
    #         n_games += 1
    #         append = '\n'

    #     print(append, sample_dict['game'], sample_dict['audio'].split('/')[-1], sample_dict['genre'])
    #     n_samples += 1

    # print(f"\nN Games: {n_games} | N Samples {n_samples}")

    run_inference(samples_dicts, model_path, base_path, gt_base_path, get_gt)

if __name__ == "__main__":
    main()