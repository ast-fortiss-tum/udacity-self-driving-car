import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
from tqdm import tqdm

import utils
from config import Config
from train_vae import setup_vae
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS
from variational_autoencoder import normalize_and_reshape, reshape

np.random.seed(0)


def plot_pictures_orig_rec(orig, dec, picture_name, loss):
    f = plt.figure(figsize=(20, 8))

    f.add_subplot(1, 2, 1)
    plt.imshow(orig.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
    plt.title("Original")

    f.add_subplot(1, 2, 2)
    plt.imshow(dec.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
    plt.title("Reconstructed loss %.4f" % loss)
    plt.show(block=True)

    if picture_name is not None:
        plt.savefig(picture_name, bbox_inches='tight')

    plt.close()


def load_or_compute_losses(anomaly_detector, encoder, decoder, dataset, cached_file_name, delete_cache):
    losses = []
    current_path = os.getcwd()
    cache_path = os.path.join(current_path, 'cache', cached_file_name)

    if delete_cache:
        if os.path.exists(cache_path):
            os.remove(cache_path)

    try:
        losses = np.load(cache_path)
        losses = losses.tolist()
        print("Found losses data for " + cached_file_name)
        return losses
    except FileNotFoundError:
        print("Losses data for " + cached_file_name + " not found. Computing...")
        i = 0
        for x in tqdm(dataset):
            i = i + 1

            x = utils.resize(x)
            x = normalize_and_reshape(x)

            import matplotlib.pyplot as plt
            plt.imshow(x.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
            plt.show()
            plt.clf()

            # TODO: the version with the VAE loss requires special treatment
            if "VAEloss" in cached_file_name:
                loss = anomaly_detector.test_on_batch(x)
            else:
                loss = anomaly_detector.test_on_batch(x, x)

            # TODO: the version with the VAE loss requires special treatment (encoder / decoder)
            encoded = encoder.predict(x)[2]
            reconstructed = decoder.predict(encoded)

            plt.imshow(reconstructed.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
            plt.show()
            plt.clf()

            # if i % 100 == 0:
            #     plot_pictures_orig_rec(x, x_rec, "picture.png", loss)

            losses.append(loss)

        # display a 2D plot of the digit classes in the latent space
        # z_mean, _, _ = encoder.predict(normalize_and_reshape(utils.resize(dataset)))
        # plt.figure(figsize=(12, 10))
        # plt.scatter(z_mean[:, 0], z_mean[:, 1])
        # plt.colorbar()
        # plt.xlabel("z[0]")
        # plt.ylabel("z[1]")
        # plt.show()

        np_losses = np.array(losses)
        np.save(cache_path, np_losses)
        print("Losses data for " + cached_file_name + " saved.")

    return losses


def load_all_images(cfg):
    """
    Load all driving images
    TODO: inefficient
    """
    drive = utils.get_driving_styles(cfg)

    print("Loading data set " + str(cfg.TRACK) + str(drive))

    start = time.time()

    x = None
    path = None

    for drive_style in drive:
        try:
            path = os.path.join(cfg.TRAINING_DATA_DIR,
                                cfg.SIMULATION_DATA_DIR,
                                cfg.TRACK,
                                drive_style,
                                'driving_log.csv')
            data_df = pd.read_csv(path)

            if x is None:
                x = data_df[['center']].values
            else:
                x = np.concatenate((x, data_df[['center']].values), axis=0)

            if cfg.TRACK == "track1":
                # print("Loading only the first 1102 images from %s (one lap)" % track)
                x = x[:cfg.TRACK1_IMG_PER_LAP]
            else:
                print("Not yet implemented! Quitting...")
                exit()

        except FileNotFoundError:
            print("Unable to read file %s" % path)
            continue

    if x is None:
        print("No driving data were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    # load the images
    images = np.empty([len(x), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

    # TODO: loading only center image for now.
    print("WARNING! Loading only front-facing images")

    for i, path in enumerate(x):
        image = utils.load_image(cfg.SIMULATION_DATA_DIR, path[0])  # load center images

        # visualize whether the input image as expected
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()

        images[i] = image

    duration_train = time.time() - start
    print("Loading data set completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    print("Data set: " + str(len(images)) + " elements")

    return images


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


def draw_best_worst_results(dataset, autoencoder, losses, picture_name, numb_of_picture=10):
    model = tensorflow.keras.models.load_model('sao/' + autoencoder.__str__())

    extremes, extremes_loss = get_best_and_worst_by_loss(dataset, losses, numb_of_picture // 2)
    # extremes = model.reshape(extremes)

    # extremes = utils.resize(extremes)
    extremes = normalize_and_reshape(extremes)

    anomaly_decoded_img = model.predict(extremes)
    plot_picture_orig_dec(extremes, anomaly_decoded_img, picture_name, extremes_loss, numb_of_picture)


def get_best_and_worst_by_loss(dataset, losses, n=5):
    loss_picture_list = []
    for idx, loss in enumerate(losses):
        picture = dataset[idx]
        loss_picture_list.append([loss, picture])

    loss_picture_list = sorted(loss_picture_list, key=lambda x: x[0])

    result = []
    losses = []
    for idx in range(0, n):
        result.append(loss_picture_list[idx][1])
        losses.append(loss_picture_list[idx][0])

    for idx in range(len(loss_picture_list) - n, len(loss_picture_list)):
        result.append(loss_picture_list[idx][1])
        losses.append(loss_picture_list[idx][0])

    return np.array(result), losses


def compute_losses_vae(cfg, name, images):
    """
    Evaluate the VAE model, compute reconstruction losses
    """

    # TODO: do not use .h5 extension when saving/loading custom objects. No longer compatible across platforms!
    my_file = Path(os.path.join(cfg.SAO_MODELS_DIR, name))

    if not my_file.exists():
        print("Model %s does not exists. Do training first." % str(name))
        return
    else:
        print("Found model %s. Loading..." % str(name))
        # TODO: do not use .h5 extension when saving/loading custom objects. No longer compatible across platforms!
        model = tensorflow.keras.models.load_model(my_file.__str__())

        encoder_name = name.replace("VAE-", "encoder-")
        encoder_file = Path(os.path.join(cfg.SAO_MODELS_DIR, encoder_name))
        encoder = tensorflow.keras.models.load_model(encoder_file.__str__())

        decoder_name = name.replace("VAE-", "decoder-")
        decoder_file = Path(os.path.join(cfg.SAO_MODELS_DIR, decoder_name))
        decoder = tensorflow.keras.models.load_model(decoder_file.__str__())

    print("Start computing reconstruction losses.")
    start = time.time()

    images = images
    losses = load_or_compute_losses(model, encoder, decoder, images, name + "-losses.npy", delete_cache=False)

    duration = time.time() - start
    print("Computed reconstruction losses completed in %s." % str(datetime.timedelta(seconds=round(duration))))

    # summarize history for loss
    plt.figure(figsize=(20, 4))
    x_losses = np.arange(len(losses))
    plt.plot(x_losses, losses, color='blue', alpha=0.7)

    plt.ylabel('Loss')
    plt.xlabel('Number of Instances')
    plt.title("Reconstruction error for " + name)

    plt.savefig('plots/reconstruction-plot-' + name + '.png')

    plt.show()

    plt.clf()
    plt.hist(losses, bins=len(losses) // 5)  # TODO: find an appropriate constant
    plt.show()

    return losses


def compute_and_plot_all_losses(cfg, use_mse=True):
    current_path = os.getcwd()
    if use_mse:
        cache_path1 = os.path.join(current_path, 'cache', "VAE-track1-MSEloss-allimg-nocrop-losses.npy")
        cache_path2 = os.path.join(current_path, 'cache', "VAE-track1-MSEloss-allimg-usecrop-losses.npy")
        cache_path3 = os.path.join(current_path, 'cache', "VAE-track1-MSEloss-centerimg-nocrop-losses.npy")
        cache_path4 = os.path.join(current_path, 'cache', "VAE-track1-MSEloss-centerimg-usecrop-losses.npy")

        losses1 = np.load(cache_path1).tolist()
        losses2 = np.load(cache_path2).tolist()
        losses3 = np.load(cache_path3).tolist()
        losses4 = np.load(cache_path4).tolist()

        # summarize history for loss
        plt.figure(figsize=(20, 4))
        x_losses = np.arange(len(losses1))
        plt.plot(x_losses, losses1, color='red', alpha=0.7, label="MSEloss-allimg-nocrop")
        plt.plot(x_losses, losses2, color='blue', alpha=0.7, label="MSEloss-allimg-usecrop")
        plt.plot(x_losses, losses3, color='green', alpha=0.7, label="MSEloss-centerimg-nocrop")
        plt.plot(x_losses, losses4, color='black', alpha=0.7, label="MSEloss-centerimg-usecrop")
    else:
        cache_path1 = os.path.join(current_path, 'cache', "VAE-track1-VAEloss-allimg-nocrop-losses.npy")
        cache_path2 = os.path.join(current_path, 'cache', "VAE-track1-VAEloss-allimg-usecrop-losses.npy")
        cache_path3 = os.path.join(current_path, 'cache', "VAE-track1-VAEloss-centerimg-nocrop-losses.npy")
        cache_path4 = os.path.join(current_path, 'cache', "VAE-track1-VAEloss-centerimg-usecrop-losses.npy")

        losses1 = np.load(cache_path1).tolist()
        losses2 = np.load(cache_path2).tolist()
        losses3 = np.load(cache_path3).tolist()
        losses4 = np.load(cache_path4).tolist()

        # summarize history for loss
        plt.figure(figsize=(20, 4))
        x_losses = np.arange(len(losses1))
        plt.plot(x_losses, losses1, color='red', alpha=0.7, label="VAEloss-allimg-nocrop")
        plt.plot(x_losses, losses2, color='blue', alpha=0.7, label="VAEloss-allimg-usecrop")
        plt.plot(x_losses, losses3, color='green', alpha=0.7, label="VAEloss-centerimg-nocrop")
        plt.plot(x_losses, losses4, color='black', alpha=0.7, label="VAEloss-centerimg-usecrop")

    plt.ylabel('Loss')
    plt.xlabel('Frames')
    plt.title("Reconstruction error for all VAE loss-based anomaly detectors")
    plt.legend()

    if use_mse:
        plt.savefig('plots/reconstruction-plot-all-MSE.png')
    else:
        plt.savefig('plots/reconstruction-plot-all-VAE.png')

    plt.show()


def load_and_eval_vae(cfg, data):
    vae, name = setup_vae(cfg)

    # history = np.load(Path(os.path.join(cfg.SAO_MODELS_DIR, name)).__str__() + ".npy", allow_pickle=True).item()
    # plot_history(history, cfg, name, vae)

    losses = compute_losses_vae(cfg, name, data)

    # compute_and_plot_all_losses(cfg, use_mse=True)

    # draw_best_worst_results(data, name, losses, "picture_name", numb_of_picture=10)


def main():
    cfg = Config()
    cfg.from_pyfile("myconfig.py")

    data = load_all_images(cfg)
    load_and_eval_vae(cfg, data)


if __name__ == '__main__':
    main()
