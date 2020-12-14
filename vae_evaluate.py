import os

import matplotlib.pyplot as plt
import numpy as np
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


# def draw_best_worst_results(dataset, autoencoder, losses, picture_name, numb_of_picture=10):
#     model = tensorflow.keras.models.load_model('sao/' + autoencoder.__str__())
#
#     extremes, extremes_loss = get_best_and_worst_by_loss(dataset, losses, numb_of_picture // 2)
#     # extremes = model.reshape(extremes)
#
#     # extremes = utils.resize(extremes)
#     extremes = normalize_and_reshape(extremes)
#
#     anomaly_decoded_img = model.predict(extremes)
#     plot_picture_orig_dec(extremes, anomaly_decoded_img, picture_name, extremes_loss, numb_of_picture)


# def get_best_and_worst_by_loss(dataset, losses, n=5):
#     loss_picture_list = []
#     for idx, loss in enumerate(losses):
#         picture = dataset[idx]
#         loss_picture_list.append([loss, picture])
#
#     loss_picture_list = sorted(loss_picture_list, key=lambda x: x[0])
#
#     result = []
#     losses = []
#     for idx in range(0, n):
#         result.append(loss_picture_list[idx][1])
#         losses.append(loss_picture_list[idx][0])
#
#     for idx in range(len(loss_picture_list) - n, len(loss_picture_list)):
#         result.append(loss_picture_list[idx][1])
#         losses.append(loss_picture_list[idx][0])
#
#     return np.array(result), losses


# def compute_and_plot_all_losses(cfg, use_mse=True):
#     current_path = os.getcwd()
#     if use_mse:
#         cache_path1 = os.path.join(current_path, 'cache', "VAE-track1-MSEloss-allimg-nocrop-losses.npy")
#         cache_path2 = os.path.join(current_path, 'cache', "VAE-track1-MSEloss-allimg-usecrop-losses.npy")
#         cache_path3 = os.path.join(current_path, 'cache', "VAE-track1-MSEloss-centerimg-nocrop-losses.npy")
#         cache_path4 = os.path.join(current_path, 'cache', "VAE-track1-MSEloss-centerimg-usecrop-losses.npy")
#
#         losses1 = np.load(cache_path1).tolist()
#         losses2 = np.load(cache_path2).tolist()
#         losses3 = np.load(cache_path3).tolist()
#         losses4 = np.load(cache_path4).tolist()
#
#         # summarize history for loss
#         plt.figure(figsize=(20, 4))
#         x_losses = np.arange(len(losses1))
#         plt.plot(x_losses, losses1, color='red', alpha=0.7, label="MSEloss-allimg-nocrop")
#         plt.plot(x_losses, losses2, color='blue', alpha=0.7, label="MSEloss-allimg-usecrop")
#         plt.plot(x_losses, losses3, color='green', alpha=0.7, label="MSEloss-centerimg-nocrop")
#         plt.plot(x_losses, losses4, color='black', alpha=0.7, label="MSEloss-centerimg-usecrop")
#     else:
#         cache_path1 = os.path.join(current_path, 'cache', "VAE-track1-VAEloss-allimg-nocrop-losses.npy")
#         cache_path2 = os.path.join(current_path, 'cache', "VAE-track1-VAEloss-allimg-usecrop-losses.npy")
#         cache_path3 = os.path.join(current_path, 'cache', "VAE-track1-VAEloss-centerimg-nocrop-losses.npy")
#         cache_path4 = os.path.join(current_path, 'cache', "VAE-track1-VAEloss-centerimg-usecrop-losses.npy")
#
#         losses1 = np.load(cache_path1).tolist()
#         losses2 = np.load(cache_path2).tolist()
#         losses3 = np.load(cache_path3).tolist()
#         losses4 = np.load(cache_path4).tolist()
#
#         # summarize history for loss
#         plt.figure(figsize=(20, 4))
#         x_losses = np.arange(len(losses1))
#         plt.plot(x_losses, losses1, color='red', alpha=0.7, label="VAEloss-allimg-nocrop")
#         plt.plot(x_losses, losses2, color='blue', alpha=0.7, label="VAEloss-allimg-usecrop")
#         plt.plot(x_losses, losses3, color='green', alpha=0.7, label="VAEloss-centerimg-nocrop")
#         plt.plot(x_losses, losses4, color='black', alpha=0.7, label="VAEloss-centerimg-usecrop")
#
#     plt.ylabel('Loss')
#     plt.xlabel('Frames')
#     plt.title("Reconstruction error for all VAE loss-based anomaly detectors")
#     plt.legend()
#
#     if use_mse:
#         plt.savefig('plots/reconstruction-plot-all-MSE.png')
#     else:
#         plt.savefig('plots/reconstruction-plot-all-VAE.png')
#
#     plt.show()


def load_and_eval_vae(cfg, dataset, delete_cache):
    vae, name = load_vae(cfg, load_vae_from_disk=True)

    # history = np.load(Path(os.path.join(cfg.SAO_MODELS_DIR, name)).__str__() + ".npy", allow_pickle=True).item()
    # plot_history(history, cfg, name, vae)

    losses = load_or_compute_losses(vae, dataset, name, delete_cache)
    plot_reconstruction_losses(losses, name)


def main():
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    dataset = load_all_images(cfg)
    load_and_eval_vae(cfg, dataset, delete_cache=True)


if __name__ == '__main__':
    main()
