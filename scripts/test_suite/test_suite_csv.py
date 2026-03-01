import os
import json
import argparse
import random
from datetime import datetime

import pandas as pd
import numpy as np

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
            audio = general_json_dict['name']

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
                    'video': choosen_audio_content['video'].split('nintendo-snes-spc/')[-1],
                    'audio': audio_name,
                    'description': choosen_audio_content['description']
                }

                choosen_samples.append(choosen_sample)

    return choosen_samples

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="/home/es119256/dados/xps/checkpoints_and_inference", help="path to folder where results will be stored")
    parser.add_argument('--genres_path', type=str, default="/home/es119256/dados/datasets/vmdb_3/deepseek_genres.csv", help="path to games genres csv")
    parser.add_argument('--split', type=str, default="test", help="split to be accessed in dataset/snes_mvdb/SPLIT")
    parser.add_argument('--converted_dataset', type=str, default="/home/es119256/dados/repos/visual-bardo-video/dataset", help="path to audiocraft/dataset. snes_mvdb will be added to access the converted dataset")

    args = parser.parse_args()

    genres_path = args.genres_path
    dataset_split_path = os.path.join(args.converted_dataset, 'snes_mvdb', args.split)

    date = datetime.now()
    date = date.strftime("%m_%d_%y")
    csv_path = os.path.join(args.base_path, f'test_suite_videos.csv_{date}',)

    print(f"dataset_split_path: {dataset_split_path}")

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

    samples_df = pd.DataFrame(samples_dicts)
    samples_df.to_csv(csv_path)

if __name__ == "__main__":
    main()