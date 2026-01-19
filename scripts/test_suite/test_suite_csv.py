########################################################################################
# To get a CSV file with the paths to the videos that are being used in the test suite
########################################################################################

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
            json_dict['video'] = general_json_dict['video'].split('nintendo-snes-spc/')[-1]
            json_dict['audio'] = general_json_dict['name']
            json_dict['description'] = general_json_dict['description']

        samples_dicts.append(json_dict)

    return samples_dicts

def get_one_sample_per_game(samples_dicts:list[dict[str, str]]) -> dict[str, list[str]]:
    # dumb dict in order to process the last game in the samples_dicts list
    none_dict = {'game': "NONE"}
    samples_dicts.append(none_dict)

    choosen_samples:dict[str, list[str]] = {
        'game': [],
        'genre': [],
        'video': [],
        'audio': [],
        'description': []
    }

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

            choosen_samples['game'].append(choosen_sample['game'])
            choosen_samples['genre'].append(choosen_sample['genre'])
            choosen_samples['video'].append(choosen_sample['video'])
            choosen_samples['audio'].append(choosen_sample['audio'])
            choosen_samples['description'].append(choosen_sample['description'])

            game_dicts.clear()

            current_game = game
            game_dicts.append(sample_dict)

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
    base_path = os.path.join(args.base_path, f'csv_{date}')
    csv_path = os.path.join(base_path, 'test_suite_videos.csv')

    if not os.path.isdir(base_path):
        os.mkdir(base_path)


    print(f"dataset_split_path: {dataset_split_path}")

    random.seed(42)

    # Read dataset split, select samples and run inference
    samples_dicts = read_dataset_split(dataset_split_path, genres_path)
    samples_dicts = get_one_sample_per_game(samples_dicts)
    samples_dicts = pd.DataFrame(samples_dicts)

    #print(samples_dicts.head())
    #print(samples_dicts.tail())

    samples_dicts.to_csv(csv_path)

if __name__ == "__main__":
    main()