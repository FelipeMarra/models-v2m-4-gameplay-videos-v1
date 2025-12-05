import os
import json
import matplotlib.pyplot as plt

XP_PATH = "/app/xps/audiocraft_gvmgen/xps/524c0212"
HISTORY_PATH = os.path.join(XP_PATH, "history_copy.json")
SAVE_PATH = os.path.join(XP_PATH, "history_copy")
METRICS = ['ce', 'ppl']

def main():
    history = None

    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)

    results = {
        key:{
            'train': [],
            'valid': []
        } for key in METRICS
    }

    for step in history:
        train = step['train']
        valid = step['valid']

        for metric in METRICS:
            results[metric]['train'].append(train[metric])
            results[metric]['valid'].append(valid[metric])

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for metric in METRICS:
        plt.rcParams["figure.figsize"] = (20,5)

        plt.plot(results[metric]['train'])
        plt.plot(results[metric]['valid'])

        plt.xticks(range(0, 130, 4))

        plt.grid()

        plt.legend(('train', 'valid'))
        plt.title(metric.upper())

        plt.savefig(f"{SAVE_PATH}/{metric}.png")
        plt.clf()

    os.rename(HISTORY_PATH, f"{SAVE_PATH}/history.json")

if __name__ == "__main__":
    main()