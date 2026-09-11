"""
    To create the files_background.cvs and files_tests.cvs and run FAD for each genre
"""

import os
import logging
import argparse
import subprocess

import csv

logger = logging.getLogger(__name__)


# Code for running the metric in a standalone fashion for each genre
GENRES = ['Action', 'Adventure', 'Fighting', 'Platform', 'Puzzle', 'RPG', 'Racing', 'Shooters', 'Simulation', 'Sports', 'Strategy']

def map_sdtk_to_genre(folder, genres_dict):
    sdtk_to_genre_dict = {genre: [] for genre in GENRES}

    paths = os.listdir(folder)
    paths = [os.path.join(folder, path) for path in paths]

    for path in paths:
        game = path.split('/')[-1].split('_')[0]
        genre = genres_dict[game]
        sdtk_to_genre_dict[genre].append(path+'\n')

    return sdtk_to_genre_dict


def write_dict(dict_to_write, file_name, save_dir):
    for genre, paths in dict_to_write.items():
        final_save_dir = os.path.join(save_dir, genre)
        os.makedirs(final_save_dir, exist_ok=True)

        file_path = os.path.join(final_save_dir, file_name)
        with open(file_path, 'w') as f:
            f.writelines(paths)


def create_files(args):
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.deepseek_genres, mode="r") as csv_file:
        genres_dict = {row["game_folder"]: row["game_genre"] for row in csv.DictReader(csv_file)}

    test_audios_folder = os.path.join(args.eval_path, 'fad/tests')
    background_audios_folder = os.path.join(args.eval_path, 'fad/background')

    test_genre_dict = map_sdtk_to_genre(test_audios_folder, genres_dict)
    background_genre_dict = map_sdtk_to_genre(background_audios_folder, genres_dict)

    write_dict(test_genre_dict, 'files_tests.cvs', save_dir=args.save_dir)
    write_dict(background_genre_dict, 'files_background.cvs', save_dir=args.save_dir)

XP = "frozen_vivit"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', type=str, default=f"./{XP}", help="path for the eval xp folder")
    parser.add_argument('--deepseek_genres', type=str, default="./deepseek_genres.csv", help="path for the snes mvdb dataset games folder")
    parser.add_argument('--save_dir', type=str, default=f"./{XP}/fad_per_genre", help="path for the snes mvdb dataset games folder")

    args = parser.parse_args()

    create_files(args)

    results_file = os.path.abspath(f'./{XP}/fad_results.csv')
    with open(results_file, 'w') as f:
        f.write("Genre,FAD_Score\n")

    for genre_folder in os.listdir(args.save_dir):
        folder_path = os.path.join(args.save_dir, genre_folder)
        test_file_cvs = os.path.abspath(os.path.join(folder_path, 'files_tests.cvs'))
        back_file_cvs = os.path.abspath(os.path.join(folder_path, 'files_background.cvs'))

        save_tests = os.path.abspath(os.path.join(folder_path, 'stats_tests'))
        save_back = os.path.abspath(os.path.join(folder_path, 'stats_background'))


        # Using a shell pipe to count lines
        cmd = f"""
            source /home/felipe/anaconda3/bin/activate 
            conda activate fad

            FAD_MODEL_PATH="./vggish_model.ckpt"
            FAD_ENV_PATH="/home/felipe/anaconda3/envs/fad"

            export TF_PYTHON_EXE="$FAD_ENV_PATH/bin/python"
            export TF_LIBRARY_PATH="$FAD_ENV_PATH/lib/python3.10/site-packages/nvidia/cudnn/lib"
            export TF_FORCE_GPU_ALLOW_GROWTH=true
            export PYTHONPATH="/home/felipe/Documents/Github/google-research-fad"

            CALC_TEST="$TF_PYTHON_EXE -m frechet_audio_distance.create_embeddings_main --model_ckpt $FAD_MODEL_PATH --input_files {test_file_cvs} --stats {save_tests} --batch_size 1"
            CALC_BACK="$TF_PYTHON_EXE -m frechet_audio_distance.create_embeddings_main --model_ckpt $FAD_MODEL_PATH --input_files {back_file_cvs} --stats {save_back} --batch_size 1"

            echo "CALC TEST"
            $CALC_TEST
            echo "CALC TEST COMPLETED"
            sleep 2

            echo "CALC BACK"
            $CALC_BACK
            echo "CALC BACK COMPLETED"
            sleep 2

            echo "CALC FAD ({genre_folder})"
            # Capture the result
            FAD_RESULT=$($TF_PYTHON_EXE -m frechet_audio_distance.compute_fad --test_stats {save_tests} --background_stats {save_back})
            sleep 2

            echo "----------------------------------------"
            echo "FAD RESULT FOR {genre_folder}: $FAD_RESULT"
            echo "----------------------------------------"

            # Save the result to the CSV file
            echo "{genre_folder},$FAD_RESULT" >> {results_file}
        """
        subprocess.run(cmd, shell=True, executable="/bin/bash")