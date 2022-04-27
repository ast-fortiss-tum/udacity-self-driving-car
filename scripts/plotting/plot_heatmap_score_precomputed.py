# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

from utils import *
from vae_evaluate import get_threshold


def score_when_decrease(output):
    return -1.0 * output[:, 0]


if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts/plotting', ''))

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    print("Script to plot score function based on heatmap")

    # summary_type = '-avg-grad'
    summary_type = '-avg'

    # 1. load heatmap scores in nominal conditions

    cfg.SIMULATION_NAME = 'gauss-journal-track1-nominal'
    path = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, 'htm-gradcam++-scores' + summary_type + '.npy')
    print(path)
    original_losses = np.load(path)
    # original_losses = original_losses[:1500]

    # retrieve a statistical threshold fitting a Gamma distribution
    threshold = get_threshold(original_losses, conf_level=0.95)

    # 2. load heatmap scores in anomalous conditions

    cfg.SIMULATION_NAME = 'ood/xai-track1-rain-10'
    path = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, 'htm-gradcam++-scores' + summary_type + '.npy')
    print(path)
    anomalous_losses = np.load(path)

    x_losses = range(0, len(anomalous_losses))
    avg_heatmaps = np.asarray(anomalous_losses)

    x_threshold = np.arange(len(avg_heatmaps))
    y_threshold = [threshold] * len(x_threshold)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)

    plt.figure(figsize=(30, 8))

    # changes the frequency of the ticks on the X-axis to simulation's seconds
    plt.xticks(
        np.arange(0, len(avg_heatmaps) + 1, cfg.FPS),
        labels=range(0, len(avg_heatmaps) // cfg.FPS + 1))
    plt.xticks(rotation=90)

    # visualize crashes
    crashes = data_df[data_df["crashed"] == 1]
    is_crash = (crashes.crashed - 1) + threshold
    plt.plot(is_crash, 'x:r', markersize=4)

    plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)
    plt.plot(x_losses, avg_heatmaps, color='blue', alpha=0.7, label='heatmap scores')

    plt.ylabel('Heatmap score')
    plt.xlabel('Frames')
    plt.title(summary_type)
    plt.legend()
    # plt.savefig("plots/avg-heatmaps.png" if summary_type is "AVERAGE" else "plots/err-diff-heatmaps.png")
    plt.show()
