from matplotlib.pyplot import xticks
from scipy.stats import gamma

from config import Config
from utils import *

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    plt.figure(figsize=(30, 8))

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)

    # read uncertainty values
    uncertainties = data_df["uncertainty"]

    WINDOW = 15
    ALPHA = 0.3
    sma = uncertainties.rolling(WINDOW, min_periods=1).mean()
    ewm = uncertainties.ewm(min_periods=1, alpha=ALPHA).mean()

    shape, loc, scale = gamma.fit(uncertainties, floc=0)
    threshold = gamma.ppf(0.68, shape, loc=loc, scale=scale)
    print(threshold)
    cfg.UNCERTAINTY_TOLERANCE_LEVEL = threshold

    x_losses = np.arange((len(uncertainties)))
    x_threshold = np.arange(len(uncertainties))
    y_threshold = [cfg.UNCERTAINTY_TOLERANCE_LEVEL] * len(x_threshold)

    # count how many mis-behaviours
    a = pd.Series(ewm)
    exc = a.ge(cfg.UNCERTAINTY_TOLERANCE_LEVEL)
    times = (exc.shift().ne(exc) & exc).sum()

    # changes the frequency of the ticks on the X-axis to simulation's seconds
    plt.xticks(
        np.arange(0, len(uncertainties) + 1, cfg.FPS),
        labels=range(0, len(uncertainties) // cfg.FPS + 1))

    plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)
    plt.plot(x_losses, uncertainties, '--', color='black', alpha=0.4, label="original")
    plt.plot(x_losses, sma,  '-.', color="blue", alpha=0.2,
             label='pred unc' + ' (sma-w' + str(WINDOW) + ')')
    plt.plot(x_losses, ewm, color="green", alpha=0.8,
             label='pred unc' + ' (ewm-a' + str(ALPHA) + ')')

    # visualize crashes
    crashes = data_df[data_df["crashed"] == 1]
    is_crash = (crashes.crashed - 1) + cfg.UNCERTAINTY_TOLERANCE_LEVEL
    plt.plot(is_crash, 'x:r', markersize=4)

    plt.legend()

    # new_x_values = range(0, len(uncertainties))
    plt.ylabel('Uncertainty')
    plt.xlabel('Frames')
    # plt.xticks(uncertainties, new_x_values)
    plt.title("Uncertainty values for "
              + cfg.SIMULATION_NAME +
              "\n# misbehaviour: %d" % times, fontsize=20)

    plt.savefig('plots/uncertainty.png')

    plt.show()
