# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import glob

from scipy.stats import gamma

from utils import *

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts/plotting', ''))

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        'gauss-journal-track1-nominal',
                        'driving_log.csv')

    data_df = pd.read_csv(path)
    all_err = data_df.loc[data_df['waypoint'] > 0]['loss']

    shape, loc, scale = gamma.fit(all_err, floc=0)
    threshold = gamma.ppf(0.95, shape, loc=loc, scale=scale)
    print(threshold)

    cfg.SIMULATION_NAME = 'xai-track1-rain-10'

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    all_err = data_df.loc[data_df['waypoint'] > 0]['loss']

    WINDOW = cfg.FPS
    ALPHA = 0.2
    sma = all_err.rolling(WINDOW, min_periods=1).mean()
    ewm = all_err.ewm(min_periods=1, alpha=ALPHA).mean()

    # count how many mis-behaviours
    a = pd.Series(ewm)
    exc = a.ge(threshold)
    times = (exc.shift().ne(exc) & exc).sum()
    times = times

    x_losses = np.arange(len(all_err))
    x_threshold = np.arange(len(all_err))
    y_threshold = [threshold] * len(x_threshold)

    plt.figure(figsize=(30, 8))

    # changes the frequency of the ticks on the X-axis to simulation's seconds
    plt.xticks(
        np.arange(0, len(all_err) + 1, cfg.FPS),
        labels=range(0, len(all_err) // cfg.FPS + 1))
    plt.xticks(rotation=90)

    # visualize crashes
    crashes = data_df[data_df["crashed"] == 1]
    is_crash = (crashes.crashed - 1) + threshold
    plt.plot(is_crash, 'x:r', markersize=4)

    plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)
    plt.plot(x_losses, all_err, '--', color="black", alpha=0.4,
             label=cfg.ANOMALY_DETECTOR_NAME + ' (original)')
    plt.plot(x_losses, sma, '-.', color="blue", alpha=0.4,
             label=cfg.ANOMALY_DETECTOR_NAME + ' (sma-w' + str(WINDOW) + ')')
    plt.plot(x_losses, ewm, color="green", alpha=0.8,
             label=cfg.ANOMALY_DETECTOR_NAME + ' (ewm-a' + str(ALPHA) + ')')

    plt.legend()
    plt.ylabel('Rec Err')
    plt.xlabel('Frames')
    plt.title("Rec Err values for "
              + cfg.SIMULATION_NAME +
              "\n# misbehaviour: %d" % times, fontsize=20)

    # plt.savefig('plots/rec-err.png')

    plt.show()
