from config import Config
from utils import *

ALL = ['centerimg-usecrop', 'centerimg-nocrop', 'allimg-usecrop', 'allimg-nocrop']
LATENT_DIM = "-latent16-"
WEATHER = "dave2-rain"

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    interval = np.arange(10, 101, step=10)

    plt.figure(figsize=(20, 4))

    for int in interval:
        path = os.path.join(cfg.TESTING_DATA_DIR,
                            "gauss-journal-track1-" + WEATHER + str(int),
                            'driving_log.csv')
        data_df = pd.read_csv(path)

        uncertainties = data_df["uncertainty"]

        x_losses = np.arange(len(uncertainties))

        x_threshold = np.arange(len(uncertainties))
        y_threshold = [0.00328] * len(x_threshold)
        plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)

        plt.plot(x_losses, uncertainties, color=plt.jet(), alpha=0.7, label="fog" + str(int))

    plt.legend()
    plt.ylabel('Uncertainty')
    plt.xlabel('Frames')
    plt.title("Uncertainty values for " + WEATHER)

    # plt.savefig('plots/reconstruction-plot-' + name + '.png')

    plt.show()
