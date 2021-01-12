import glob

from scipy.stats import gamma

from config import Config
from utils import *

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

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

    # encoder = tensorflow.keras.models.load_model('sao/encoder-' + cfg.ANOMALY_DETECTOR_NAME)
    # decoder = tensorflow.keras.models.load_model('sao/decoder-' + cfg.ANOMALY_DETECTOR_NAME)
    #
    # vae = VAE(model_name=cfg.ANOMALY_DETECTOR_NAME,
    #           loss=cfg.LOSS_SAO_MODEL,
    #           latent_dim=cfg.SAO_LATENT_DIM, encoder=encoder, decoder=decoder)
    # vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE))

    # for img in tqdm(all_imgs):
    #     img = mpimg.imread(img)
    #     img = utils.resize(img)
    #     img = normalize_and_reshape(img)
    #
    #     rec_err = vae.test_on_batch(img)[1]
    #     all_err.append(rec_err)

    # threshold = 320  # from nominal
    shape, loc, scale = gamma.fit(all_err, floc=0)
    threshold = gamma.ppf(0.95, shape, loc=loc, scale=scale)
    print(threshold)

    # count how many mis-behaviours
    a = pd.Series(all_err)
    exc = a.ge(threshold)
    times = (exc.shift().ne(exc) & exc).sum()

    x_losses = np.arange(len(all_err))
    x_threshold = np.arange(len(all_err))
    y_threshold = [threshold] * len(x_threshold)

    # changes the frequency of the ticks on the X-axis to simulation's seconds
    plt.xticks(
        np.arange(0, len(all_err) + 1, cfg.FPS),
        labels=range(0, len(all_err) // cfg.FPS + 1))

    # visualize crashes
    crashes = data_df[data_df["crashed"] == 1]
    is_crash = (crashes.crashed - 1) + cfg.UNCERTAINTY_TOLERANCE_LEVEL
    plt.plot(is_crash, 'x:r', markersize=4)

    plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)
    plt.plot(x_losses, all_err, color="blue", alpha=0.7, label=cfg.ANOMALY_DETECTOR_NAME)

    plt.legend()
    plt.ylabel('Rec Err')
    plt.xlabel('Frames')
    plt.title("Rec Err values for "
              + cfg.SIMULATION_NAME +
              "\n# misbehaviour: %d" % times, fontsize=20)

    # plt.savefig('plots/reconstruction-plot-' + WEATHER + str(int) + '.png')

    plt.show()
