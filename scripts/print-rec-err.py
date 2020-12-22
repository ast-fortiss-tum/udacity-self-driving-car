import glob

import tensorflow
from tensorflow import keras
from tqdm import tqdm

import utils
from config import Config
from utils import *
from vae import normalize_and_reshape, VAE

ALL = ['centerimg-usecrop', 'centerimg-nocrop', 'allimg-usecrop', 'allimg-nocrop']
LATENT_DIM = "-latent16-"
WEATHER = "dave2-fog"
interval = np.arange(10, 101, step=10)

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    for int in interval:
        plt.figure(figsize=(20, 4))

        path = os.path.join(cfg.TESTING_DATA_DIR,
                            "gauss-journal-track1-" + WEATHER + str(int),
                            'IMG')
        all_imgs = glob.glob(path + "/*.jpg")
        all_err = []

        encoder = tensorflow.keras.models.load_model('sao/encoder-track1-MSEloss-intdim512-latent16-allimg-nocrop')
        decoder = tensorflow.keras.models.load_model('sao/decoder-track1-MSEloss-intdim512-latent16-allimg-nocrop')

        vae = VAE(model_name='track1-MSEloss-intdim512-latent16-allimg-nocrop', loss=cfg.LOSS_SAO_MODEL,
                  intermediate_dim=cfg.SAO_INTERMEDIATE_DIM,
                  latent_dim=cfg.SAO_LATENT_DIM, encoder=encoder, decoder=decoder)
        vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

        for img in tqdm(all_imgs):
            img = mpimg.imread(img)
            img = utils.resize(img)
            img = normalize_and_reshape(img)

            rec_err = vae.test_on_batch(img)[2]
            all_err.append(rec_err)

        threshold = 320  # from nominal

        x_losses = np.arange(len(all_err))
        x_threshold = np.arange(len(all_err))
        y_threshold = [threshold] * len(x_threshold)
        plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)

        plt.plot(x_losses, all_err, color="blue", alpha=0.7, label=WEATHER + str(int))

        plt.legend()
        plt.ylabel('Rec Err')
        plt.xlabel('Frames')
        plt.title("Rec Err values for " + WEATHER + str(int))

        plt.savefig('plots/reconstruction-plot-' + WEATHER + str(int) + '.png')

        plt.show()
