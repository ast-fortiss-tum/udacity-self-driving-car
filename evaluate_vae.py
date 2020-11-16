import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils
from config import Config
from train_vae import setup_vae
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
from vae_batch_generator import Generator
from variational_autoencoder import VariationalAutoencoder, normalize_and_reshape
from tqdm import tqdm

np.random.seed(0)


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
    except FileNotFoundError:
        print("Losses data for " + cached_file_name + " not founded. Computing...")
        for x in tqdm(dataset):
            x = utils.resize(x)
            x = normalize_and_reshape(x)
            # x = np.expand_dims(x, axis=0)
            loss = anomaly_detector.test_on_batch(x, x)
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
    tracks = cfg.TRACK
    drive = utils.get_driving_styles(cfg)

    print("Loading data set " + str(tracks) + str(drive))

    start = time.time()

    x = None
    path = None

    for track in tracks:
        for drive_style in drive:
            try:
                path = os.path.join(cfg.TRAINING_DATA_DIR, cfg.SIMULATION_DATA_DIR, track, drive_style,
                                    'driving_log.csv')
                data_df = pd.read_csv(path)

                if x is None:
                    x = data_df[['center']].values
                else:
                    x = np.concatenate((x, data_df[['center']].values), axis=0)

                if track == "track1":
                    # print("Loading only the first 1102 images from %s (one lap)" % track)
                    x = x[:1102]
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


def compute_losses_vae(cfg, vae, name, images):
    """
    Evaluate the VAE model, compute reconstruction losses
    """

    my_file = Path(os.path.join(cfg.SAO_MODELS_DIR, name) + '.h5')
    if not my_file.exists():
        print("Model %s does not exists. Do training first." % str(name))
        return

    print("Start computing reconstruction losses.")
    start = time.time()

    model = vae.create_autoencoder()

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


def load_and_eval_vae(cfg, data):
    vae, name = setup_vae(cfg)
    compute_losses_vae(cfg, vae, name, data)


def main():
    cfg = Config()
    cfg.from_pyfile("myconfig.py")

    data = load_all_images(cfg)
    load_and_eval_vae(cfg, data)


if __name__ == '__main__':
    main()
