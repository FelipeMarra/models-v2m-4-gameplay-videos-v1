import os
import json

EGS_PATH = "/app/code/egs/snes_mvdb"

def get_splits_durations():
    durations_dict = {}

    for split in os.listdir(EGS_PATH):
        durations_dict[split] = 0
        data_json_path = os.path.join(EGS_PATH, split, "data.jsonl")

        with open(data_json_path, 'r') as f:
            data_json_lines = f.readlines()

            for line in data_json_lines:
                audio_info = json.loads(line)

                is_json_zero = audio_info["json_path"].split('.')[0].split('_json_')[1] == "0000"

                if is_json_zero:
                    durations_dict[split] += audio_info['duration']

    return durations_dict

def main():
    durations_dict = get_splits_durations()

    print(durations_dict)

    for key, value in durations_dict.items():
        print(f"Split: {key} | Duration secs: {value}; min: {value/60}; hrs: {value/(60*60)}")

if __name__ == "__main__":
    main()