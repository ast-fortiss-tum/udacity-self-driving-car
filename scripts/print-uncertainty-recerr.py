from scipy.stats import gamma

from config import Config
from utils import *

ALL = ['centerimg-usecrop', 'centerimg-nocrop', 'allimg-usecrop', 'allimg-nocrop']
LATENT_DIM = "-latent16-"
WEATHER = "dave2-fog"

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    interval = np.arange(10, 101, step=10)

    for int in interval:
        path = os.path.join(cfg.TESTING_DATA_DIR,
                            "gauss-journal-track1-" + WEATHER + str(int),
                            'driving_log.csv')
        data_df = pd.read_csv(path)

        uncertainties = data_df["uncertainty"]
        shape, loc, scale = gamma.fit(uncertainties, floc=0)
        threshold = gamma.ppf(0.95, shape, loc=loc, scale=scale)
        uncertainties = uncertainties[uncertainties < threshold]  # 0.00328

        loss = data_df["loss"]
        shape, loc, scale = gamma.fit(loss, floc=0)
        threshold = gamma.ppf(0.95, shape, loc=loc, scale=scale)
        loss = loss[loss > threshold]

        plt.figure(figsize=(20, 4))

        x_uncertainties = np.arange(len(uncertainties))
        x_losses = np.arange(len(loss))

        # x_threshold = np.arange(len(uncertainties))
        # y_threshold = [0.00328] * len(x_threshold)
        # plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)

        plt.plot(x_uncertainties, uncertainties, color="red", alpha=0.7, label="unc")
        plt.plot(x_losses, loss, color="black", alpha=0.7, label="rec_loss")

        plt.legend()
        plt.ylabel('Uncertainty')
        plt.xlabel('Frames')
        plt.title("Uncertainty vs Rec Loss values for " + WEATHER + str(int))

        # plt.savefig('plots/reconstruction-plot-' + name + '.png')

        plt.show()
