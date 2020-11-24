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
from variational_autoencoder import normalize_and_reshape

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

    plt.savefig(picture_name, bbox_inches='tight')
    plt.close()


def load_or_compute_losses(anomaly_detector, dataset, cached_file_name, delete_cache):
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

            # TODO: the version with the VAE loss requires special treatment
            loss = anomaly_detector.test_on_batch(x)

            x_rec = anomaly_detector.predict(x)

            import matplotlib.pyplot as plt
            plt.imshow(anomaly_detector.predict_on_batch(x).reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
            plt.show()

            if i % 100 == 0:
                plot_pictures_orig_rec(x, x_rec, "picture.png", loss)

            losses.append(loss)
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

    my_file = Path(os.path.join(cfg.SAO_MODELS_DIR, name))
    if not my_file.exists():
        print("Model %s does not exists. Do training first." % str(name))
        return
    else:
        print("Found model %s. Loading..." % str(name))
        model = tensorflow.keras.models.load_model(my_file.__str__())

    print("Start computing reconstruction losses.")
    start = time.time()

    images = images
    losses = load_or_compute_losses(model, images, name + "-losses.npy", False)

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


def load_and_eval_vae(cfg, data):
    vae, name = setup_vae(cfg)
    losses = compute_losses_vae(cfg, name, data)
    draw_best_worst_results(data, name, losses, "picture_name", numb_of_picture=10)


def main():
    cfg = Config()
    cfg.from_pyfile("myconfig.py")

    data = load_all_images(cfg)
    load_and_eval_vae(cfg, data)


if __name__ == '__main__':
    main()
