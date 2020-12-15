import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma
from tqdm import tqdm

import utils
from config import Config
from utils import load_all_images
from utils import plot_reconstruction_losses
from vae import normalize_and_reshape, RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS
from vae_train import load_vae

np.random.seed(0)


def load_or_compute_losses(anomaly_detector, dataset, cached_file_name, delete_cache):
    losses = []
    current_path = os.getcwd()
    cache_path = os.path.join(current_path, 'cache', cached_file_name + '.npy')

    if delete_cache:
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print("delete_cache=true. Removed losses cache file " + cached_file_name)

    try:
        losses = np.load(cache_path)
        losses = losses.tolist()
        print("Found losses data for " + cached_file_name)
        return losses
    except FileNotFoundError:
        print("Losses data for " + cached_file_name + " not found. Computing...")

        for x in tqdm(dataset):
            x = utils.resize(x)
            x = normalize_and_reshape(x)

            # sanity check
            # z_mean, z_log_var, z = anomaly_detector.encoder.predict(x)
            # decoded = anomaly_detector.decoder.predict(z)
            # reconstructed = anomaly_detector.predict(x)

            # TODO: check the index
            loss = anomaly_detector.test_on_batch(x)[1]  # total loss
            losses.append(loss)

        np_losses = np.array(losses)
        np.save(cache_path, np_losses)
        print("Losses data for " + cached_file_name + " saved.")

    return losses


def plot_picture_orig_dec(orig, dec, picture_name, losses, num=10):
    n = num
    plt.figure(figsize=(40, 8))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(orig[i].reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title("Original Photo")

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(dec[i].reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title("Reconstructed loss %.4f" % losses[i])

    plt.savefig(picture_name, bbox_inches='tight')
    plt.show()
    plt.close()


def get_thresholds(losses):
    print("Fitting reconstruction error distribution using Gamma distribution")

    shape, loc, scale = gamma.fit(losses, floc=0)

    thresholds = []

    conf_intervals = [0.95]  # [0.95, 0.99, 0.999, 0.9999, 0.99999]

    print("Creating thresholds using the confidence intervals: %s" % conf_intervals)

    for c in conf_intervals:
        thresholds.append(gamma.ppf(c, shape, loc=loc, scale=scale))

    return thresholds


def load_and_eval_vae(cfg, dataset, delete_cache):
    vae, name = load_vae(cfg, load_vae_from_disk=True)

    # history = np.load(Path(os.path.join(cfg.SAO_MODELS_DIR, name)).__str__() + ".npy", allow_pickle=True).item()
    # plot_history(history, cfg, name, vae)

    losses = load_or_compute_losses(vae, dataset, name, delete_cache)
    thresholds = get_thresholds(losses)
    plot_reconstruction_losses(losses, name, thresholds)


def main():
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    dataset = load_all_images(cfg)
    load_and_eval_vae(cfg, dataset, delete_cache=False)


if __name__ == '__main__':
    main()
