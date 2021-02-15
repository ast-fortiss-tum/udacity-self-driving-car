import glob

from scipy.stats import gamma

from config import Config
from utils import *

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    cfg.SIMULATION_NAME = 'gauss-journal-track1-rain'

    plt.figure(figsize=(30, 8))

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'IMG')

    all_imgs = glob.glob(path + "/*.jpg")

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    all_err = data_df['loss']

    WINDOW = 15
    ALPHA = 0.2
    sma = all_err.rolling(WINDOW, min_periods=1).mean()
    ewm = all_err.ewm(min_periods=1, alpha=ALPHA).mean()

    shape, loc, scale = gamma.fit(all_err, floc=0)
    threshold = gamma.ppf(0.68, shape, loc=loc, scale=scale)
    print(threshold)

    # count how many mis-behaviours
    a = pd.Series(ewm)
    exc = a.ge(threshold)
    times = (exc.shift().ne(exc) & exc).sum()
    times = times

    x_losses = np.arange(len(all_err))
    x_threshold = np.arange(len(all_err))
    y_threshold = [threshold] * len(x_threshold)

    # changes the frequency of the ticks on the X-axis to simulation's seconds
    plt.xticks(
        np.arange(0, len(all_err) + 1, cfg.FPS),
        labels=range(0, len(all_err) // cfg.FPS + 1))

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
