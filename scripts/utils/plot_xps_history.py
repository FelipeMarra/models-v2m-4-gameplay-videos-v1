import os
import json
import matplotlib.pyplot as plt

XP_PATH = "/home/es119256/dados/xps/audiocraft_vivit_felipe/xps/fake/"
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

        plt.xticks(range(0, 119, 4))

        plt.grid()

        plt.legend(('train', 'valid'))
        plt.title(metric.upper())

        plt.savefig(f"{SAVE_PATH}/{metric}.png")
        plt.clf()

    # os.rename(HISTORY_PATH, f"{SAVE_PATH}/history.json")

    metric = 'ce'
    indexes = range(len(results[metric]['valid']))
    indexes = sorted(indexes, key=lambda x: results[metric]['valid'][x])
    print(f"Best Valid Epoch: {indexes[0]+1}, with value {results[metric]['valid'][indexes[0]]}")
    print(f"Worst Valid Epoch: {indexes[-1]+1}, with value {results[metric]['valid'][indexes[-1]]}")

if __name__ == "__main__":
    main()