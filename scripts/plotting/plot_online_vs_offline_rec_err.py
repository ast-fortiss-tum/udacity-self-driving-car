# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
from scipy.stats import stats

import utils
from selforacle.vae import normalize_and_reshape, VAE
from utils import *

ANOMALY_DETECTOR = "track1-MSEloss-latent2"

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts/plotting', ''))

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    print("Script to compare offline vs online (within Udacity's) reconstruction errors")

    # load the online rec errors from csv
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    online_losses = data_df["loss"]
    sim_time = np.max(data_df["time"])
    print("loaded %d losses (%d frame/s)" % (len(online_losses), len(online_losses) // sim_time))

    # compute the offline rec errors from the images stored on the fs
    data = data_df["center"]
    print("read %d images from file" % len(data))

    encoder_mse = tensorflow.keras.models.load_model('sao/encoder-' + ANOMALY_DETECTOR)
    decoder_mse = tensorflow.keras.models.load_model('sao/decoder-' + ANOMALY_DETECTOR)
    vae = VAE(model_name="encoder_mse", loss="MSE",
              latent_dim=cfg.SAO_LATENT_DIM, encoder=encoder_mse, decoder=decoder_mse)
    vae.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001))

    offline_reconstruction_errors = []
    total_time = 0

    for x in data:
        x = mpimg.imread(x)

        start = time.time()

        x = utils.resize(x)
        x = normalize_and_reshape(x)
        err = vae.test_on_batch(x)[2]
        duration = time.time() - start

        # print("Prediction completed in %s." % str(duration))
        total_time += duration

        offline_reconstruction_errors.append(err)

    print("All reconstruction completed in %s (%s/s)." % (
        str(total_time), str(total_time / len(online_losses))))

    # compute and plot the rec errors
    x_losses = np.arange(len(online_losses))
    plt.plot(x_losses, online_losses, color='blue', alpha=0.7, label='online losses')
    plt.plot(x_losses, offline_reconstruction_errors, color='green', alpha=0.7, label='offline losses')

    plt.ylabel('Loss')
    plt.xlabel('Frames')
    plt.title("offline vs online " + ANOMALY_DETECTOR)
    plt.legend()
    plt.savefig("plots/rec-err-diff.png")
    plt.show()

    print(stats.mannwhitneyu(online_losses, offline_reconstruction_errors))
