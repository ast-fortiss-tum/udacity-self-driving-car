from config import Config
from utils import *

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    interval = np.arange(10, 101, step=10)

    plt.figure(figsize=(30, 8))

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)

    # read CTE values
    cte_values = data_df["cte"]

    # apply time-series analysis over 1s
    WINDOW = 15
    ALPHA = 0.05
    sma = cte_values.rolling(WINDOW, min_periods=1).mean()
    ewm = cte_values.ewm(min_periods=1, alpha=ALPHA).mean()

    # read CTE values
    crashes = data_df[data_df["crashed"] == 1]
    is_crash = (crashes.crashed - 1) + cfg.CTE_TOLERANCE_LEVEL
    is_crash_2 = (crashes.crashed - 1) - cfg.CTE_TOLERANCE_LEVEL

    x_losses = np.arange(len(cte_values))

    x_threshold = np.arange(len(cte_values))
    y_threshold = [cfg.CTE_TOLERANCE_LEVEL] * len(x_threshold)
    y_threshold_2 = [-cfg.CTE_TOLERANCE_LEVEL] * len(x_threshold)

    # count how many mis-behaviours
    a = pd.Series(ewm)
    exc = a.ge(cfg.CTE_TOLERANCE_LEVEL)
    times_above = (exc.shift().ne(exc) & exc).sum()

    exc = a.le(-cfg.CTE_TOLERANCE_LEVEL)
    times_below = (exc.shift().le(exc) & exc).sum()

    times = times_above + times_below

    # changes the frequency of the ticks on the X-axis to simulation's seconds
    plt.xticks(
        np.arange(0, len(cte_values) + 1, cfg.FPS),
        labels=range(0, len(cte_values) // cfg.FPS + 1))

    plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)
    plt.plot(x_threshold, y_threshold_2, color='red', alpha=0.2)
    plt.plot(x_losses, cte_values, '--', color='black', alpha=0.4, label="cte")
    plt.plot(is_crash, 'x:r', markersize=4)
    plt.plot(is_crash_2, 'x:r', markersize=4)

    plt.plot(x_losses, sma, '-.', color="blue", alpha=0.4,
             label='cte' + ' (sma-w' + str(WINDOW) + ')')
    plt.plot(x_losses, ewm, color="green", alpha=0.8,
             label='cte' + ' (ewm-a' + str(ALPHA) + ')')

    plt.legend()
    plt.ylabel('CTE')
    plt.xlabel('Frames')
    plt.title("CTE values for "
              + cfg.SIMULATION_NAME +
              "\n# misbehaviour: %d (%d right, %d left)" % (times, times_above, times_below),
              fontsize=20)

    plt.savefig('plots/cte.png')

    plt.show()
