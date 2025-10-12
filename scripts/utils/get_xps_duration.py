import os
import json
import datetime

XP_PATH = "/home/felipe/Desktop/xps/xps/a1499177"
SAVE_PATH = os.path.join(XP_PATH, "history", "history.json")
METRICS = ['duration']

def main():
    history = None

    with open(SAVE_PATH, 'r') as f:
        history = json.load(f)

    duration = 0

    for step in history:
        train = step['train']
        valid = step['valid']

        duration += train['duration'] + valid['duration']

    print(str(datetime.timedelta(seconds = duration)))
    print((duration/(60*60)), "hrs")


if __name__ == "__main__":
    main()