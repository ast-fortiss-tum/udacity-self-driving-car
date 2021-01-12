from matplotlib.pyplot import xticks

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

    # apply time-series analysis over 1s
    # new_losses = []
    # temp = []
    # for idx, loss in enumerate(uncertainties):
    #     temp.append(loss)
    #     if idx is not 0 and idx % cfg.FPS == 0:
    #         new_losses.append(np.mean(temp))
    #         temp = []
    #
    # uncertainties = new_losses

    # also these two works, but do interpolate as well
    # uncertainties_ewm = data_df["uncertainty"].ewm(span=15).mean()
    uncertainties = data_df["uncertainty"].rolling(15).mean()

    x_losses = np.arange((len(uncertainties)))
    x_threshold = np.arange(len(uncertainties))
    y_threshold = [cfg.UNCERTAINTY_TOLERANCE_LEVEL] * len(x_threshold)

    # count how many mis-behaviours
    a = pd.Series(uncertainties)
    exc = a.ge(cfg.UNCERTAINTY_TOLERANCE_LEVEL)
    times = (exc.shift().ne(exc) & exc).sum()

    # changes the frequency of the ticks on the X-axis to simulation's seconds
    plt.xticks(
        np.arange(0, len(uncertainties) + 1, cfg.FPS),
        labels=range(0, len(uncertainties) // cfg.FPS + 1))

    plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)
    plt.plot(x_losses, uncertainties, color=plt.jet(), alpha=0.7, label="predictive uncertainty")

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

    # plt.savefig('plots/reconstruction-plot-' + name + '.png')

    plt.show()
