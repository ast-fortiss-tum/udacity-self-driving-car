# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
from pathlib import Path

from scipy.stats import gamma
from tf_keras_vis.saliency import Saliency
from tqdm import tqdm

import utils
from utils import *


def score_when_decrease(output):
    return -1.0 * output[:, 0]


if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    HEATMAP_SUMMARIZATION_MODE = "ERROR_DIFF"  # options: AVERAGE, ERROR_DIFF

    print("Script to plot score function based on heatmap")

    # load the image file paths from csv
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    data = data_df["center"]

    print("read %d images from file" % len(data))

    # load self-driving car model
    dave2 = tensorflow.keras.models.load_model(Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)))

    # load heatmap model
    saliency = Saliency(dave2, model_modifier=None)

    avg_heatmaps = []
    total_time = 0
    prev_hm = diff = np.zeros((80, 160))

    for idx, x in enumerate(tqdm(data)):
        err = None
        x = mpimg.imread(x)

        start_time = time.time()

        x = utils.resize(x).astype('float32')

        saliency_map = saliency(score_when_decrease, x, smooth_samples=20, smooth_noise=0.20)

        if HEATMAP_SUMMARIZATION_MODE is "AVERAGE":
            err = np.average(saliency_map)
        elif HEATMAP_SUMMARIZATION_MODE is "ERROR_DIFF":
            if idx == 0:
                diff = 0
            else:
                diff = abs(prev_hm - saliency_map)
            err = np.average(diff)
            prev_hm = saliency_map
        else:
            print("HEATMAP_SUMMARIZATION_MODE unknown. Exit.")
            exit(1)

        # DEBUG: visualize one heatmap and heatmap diff every 1000th frame
        if idx != 0 and idx % 1000 == 0:
            plt.imshow(np.squeeze(saliency_map), cmap='jet')
            plt.title("saliency map %d" % idx)
            plt.show()

            plt.clf()
            plt.imshow(np.squeeze(diff), cmap='jet')
            plt.title("difference map (%d-%d)" % (idx - 1, idx))
            plt.show()

        duration = time.time() - start_time

        # print("Prediction completed in %s." % str(duration))
        total_time += duration

        avg_heatmaps.append(err)

    print("All heatmaps completed in %s (%s/s)." % (
        str(total_time), str(total_time / len(avg_heatmaps))))

    # retrieve a statistical threshold fitting a Gamma distribution
    shape, loc, scale = gamma.fit(avg_heatmaps, floc=0)
    threshold = gamma.ppf(0.95, shape, loc=loc, scale=scale)

    x_losses = range(0, len(avg_heatmaps))
    avg_heatmaps = np.asarray(avg_heatmaps)

    x_threshold = np.arange(len(avg_heatmaps))
    y_threshold = [threshold] * len(x_threshold)

    plt.figure(figsize=(30, 8))

    # changes the frequency of the ticks on the X-axis to simulation's seconds
    plt.xticks(
        np.arange(0, len(avg_heatmaps) + 1, cfg.FPS),
        labels=range(0, len(avg_heatmaps) // cfg.FPS + 1))
    plt.xticks(rotation=90)

    # visualize crashes
    crashes = data_df[data_df["crashed"] == 1]
    # crashes = crashes[start:end]
    is_crash = (crashes.crashed - 1) + threshold
    plt.plot(is_crash, 'x:r', markersize=4)

    plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)
    plt.plot(x_losses, avg_heatmaps, color='blue', alpha=0.7, label='avg heatmaps')

    plt.ylabel('Heatmap score')
    plt.xlabel('Frames')
    plt.title(HEATMAP_SUMMARIZATION_MODE)
    plt.legend()
    plt.savefig("plots/avg-heatmaps.png" if HEATMAP_SUMMARIZATION_MODE is "AVERAGE" else "plots/err-diff-heatmaps.png")
    plt.show()
