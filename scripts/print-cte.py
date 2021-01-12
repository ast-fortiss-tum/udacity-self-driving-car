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

    # read CTE values
    crashes = data_df[data_df["crashed"] == 1]
    is_crash = (crashes.crashed - 1) + cfg.CTE_TOLERANCE_LEVEL

    x_losses = np.arange(len(cte_values))

    x_threshold = np.arange(len(cte_values))
    y_threshold = [cfg.CTE_TOLERANCE_LEVEL] * len(x_threshold)
    y_threshold_2 = [-cfg.CTE_TOLERANCE_LEVEL] * len(x_threshold)

    # count how many mis-behaviours
    a = pd.Series(cte_values)
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
    plt.plot(x_losses, cte_values, color=plt.jet(), alpha=0.7, label="cte")
    plt.plot(is_crash, 'bo', markersize=2)

    plt.legend()
    plt.ylabel('CTE')
    plt.xlabel('Frames')
    plt.title("CTE values for "
              + cfg.SIMULATION_NAME +
              "\n# misbehaviour: %d (%d right, %d left)" % (times, times_above, times_below),
              fontsize=20)

    # plt.savefig('plots/reconstruction-plot-' + name + '.png')

    plt.show()
