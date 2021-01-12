from config import Config
from utils import *

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    interval = np.arange(10, 101, step=10)

    plt.figure(figsize=(20, 4))

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)

    uncertainties = data_df["uncertainty"]
    uncertainties = np.convolve(uncertainties, np.ones(15), 'valid') / 15

    x_losses = np.arange(len(uncertainties))

    x_threshold = np.arange(len(uncertainties))
    y_threshold = [0.00328] * len(x_threshold)

    crashes = data_df[data_df["crashed"] == 1]
    is_crash = (crashes.crashed - 1) + 0.00328

    plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)
    plt.plot(x_losses, uncertainties, color=plt.jet(), alpha=0.7, label="predictive uncertainty")
    plt.plot(is_crash, 'bo', markersize=2)

    plt.legend()
    plt.ylabel('Uncertainty')
    plt.xlabel('Frames')
    plt.title("Uncertainty values for " + cfg.SIMULATION_NAME)

    # plt.savefig('plots/reconstruction-plot-' + name + '.png')

    plt.show()
