import glob

import tensorflow
from scipy.stats import gamma
from tensorflow import keras

from config import Config
from utils import *
from vae import VAE

ALL = ['centerimg-usecrop', 'centerimg-nocrop', 'allimg-usecrop', 'allimg-nocrop']
LATENT_DIM = "-latent16-"
WEATHER = "dave2-cone-emergency-brake"

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    plt.figure(figsize=(20, 4))

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        "gauss-journal-track1-" + WEATHER,
                        'IMG')
    all_imgs = glob.glob(path + "/*.jpg")
    all_err = []

    data_df = data_df = pd.read_csv(os.path.join(cfg.TESTING_DATA_DIR,
                                                 "gauss-journal-track1-" + WEATHER,
                                                 'driving_log.csv'))
    losses = data_df["loss"]

    encoder = tensorflow.keras.models.load_model('sao/encoder-track1-MSEloss-intdim512-latent16-centerimg-nocrop')
    decoder = tensorflow.keras.models.load_model('sao/decoder-track1-MSEloss-intdim512-latent16-centerimg-nocrop')

    vae = VAE(model_name='track1-MSEloss-intdim512-latent16-allimg-nocrop', loss=cfg.LOSS_SAO_MODEL,
              intermediate_dim=cfg.SAO_INTERMEDIATE_DIM,
              latent_dim=cfg.SAO_LATENT_DIM, encoder=encoder, decoder=decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    # for i, img in enumerate(all_imgs):
    #     img = mpimg.imread(img)
    #     plt.imshow(img)
    #     plt.title(losses[i])
    #     plt.show()
    # img = mpimg.imread(img)
    # img = utils.resize(img)
    # img = normalize_and_reshape(img)
    #
    # rec_err = vae.test_on_batch(img, img)[2]
    # all_err.append(rec_err)

    # exit()

    shape, loc, scale = gamma.fit(losses[4:150], floc=0)
    threshold = gamma.ppf(0.95, shape, loc=loc, scale=scale)
    print(threshold)

    x_losses = np.arange(len(losses))
    x_threshold = np.arange(len(losses))
    y_threshold = [threshold] * len(x_threshold)

    plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)
    plt.plot(172, losses[172], 'ro')
    # plt.plot(x_losses, all_err, color="blue", alpha=0.7, label='offline rec err')
    plt.plot(np.arange(len(losses)), losses, color="green", alpha=0.7, label='in-field rec err')

    tick_locs = np.arange(211, step=15)
    tick_lbls = np.arange(15)
    plt.xticks(tick_locs, tick_lbls)

    plt.legend()
    plt.ylabel('Rec Err')
    plt.xlabel('Frames')
    plt.title("Rec Err values for " + WEATHER)

    # plt.savefig('plots/reconstruction-plot-' + WEATHER + '.png')

    plt.show()
