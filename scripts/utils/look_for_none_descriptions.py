import os
import json

DATASET_ROOT = "/app/code/dataset/snes_mvdb"

def main():
    for split in sorted(os.listdir(DATASET_ROOT)):
        split_path = os.path.join(DATASET_ROOT, split)

        for file in sorted(os.listdir(split_path)):
            if file.endswith('.mp3'):
                continue

            file_path = os.path.join(split_path, file)

            with open(file_path, 'r') as f:
                file_dict = json.load(f)
                if file_dict['description'] == "":
                    print(f"{split}/{file}")

if __name__ == "__main__":
    main()