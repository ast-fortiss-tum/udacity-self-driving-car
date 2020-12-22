import datetime
import os
import time

import numpy as np
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow import keras

from utils import RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS, get_driving_styles
from vae import Encoder, Decoder, VAE


def load_vae(cfg, load_vae_from_disk):
    if cfg.USE_ONLY_CENTER_IMG:
        print("cfg.USE_ONLY_CENTER_IMG = "
              + str(cfg.USE_ONLY_CENTER_IMG)
              + ". Using only front-facing camera images")
        use_center = '-centerimg-'
    else:
        print("cfg.USE_ONLY_CENTER_IMG = "
              + str(cfg.USE_ONLY_CENTER_IMG)
              + ". Using all camera images")
        use_center = '-allimg-'

    if cfg.USE_CROP:
        print("cfg.USE_CROP = "
              + str(cfg.USE_CROP)
              + ". Cropping the image")
        use_crop = 'usecrop'
    else:
        print("cfg.USE_CROP = "
              + str(cfg.USE_CROP)
              + ". Using the entire image")
        use_crop = 'nocrop'

    name = cfg.TRACK + '-' + cfg.LOSS_SAO_MODEL + 'loss' + "-latent" + str(cfg.SAO_LATENT_DIM) + use_center + use_crop

    if "RETRAINED" in cfg.ANOMALY_DETECTOR_NAME:
        name = name + '-RETRAINED'

    if load_vae_from_disk:
        encoder = tensorflow.keras.models.load_model('sao/' + 'encoder-' + name)
        decoder = tensorflow.keras.models.load_model('sao/' + 'decoder-' + name)
        print("loaded VAE from disk")
    else:
        encoder = Encoder().call(cfg.SAO_INTERMEDIATE_DIM, cfg.SAO_LATENT_DIM,
                                 RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS)
        decoder = Decoder().call(cfg.SAO_INTERMEDIATE_DIM,
                                 cfg.SAO_LATENT_DIM,
                                 (cfg.SAO_LATENT_DIM,), )
        print("created new VAE model from disk")

    vae = VAE(model_name=cfg.ANOMALY_DETECTOR_NAME,
              loss=cfg.LOSS_SAO_MODEL,
              latent_dim=cfg.SAO_LATENT_DIM,
              intermediate_dim=cfg.SAO_INTERMEDIATE_DIM,
              encoder=encoder,
              decoder=decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    return vae, name


# TODO: unify with load_data
def load_data_for_vae_training(cfg):
    """
    Load training data and split it into training and validation set
    """
    drive = get_driving_styles(cfg)

    print("Loading training set " + str(cfg.TRACK) + str(drive))

    start = time.time()

    x = None
    path = None
    x_train = None
    x_test = None

    for drive_style in drive:
        try:
            path = os.path.join(cfg.TRAINING_DATA_DIR,
                                cfg.TRAINING_SET_DIR,
                                cfg.TRACK,
                                drive_style,
                                'driving_log.csv')
            data_df = pd.read_csv(path)
            if x is None:
                if cfg.USE_ONLY_CENTER_IMG:
                    x = data_df[['center']].values
                else:
                    x = data_df[['center', 'left', 'right']].values
            else:
                if cfg.USE_ONLY_CENTER_IMG:
                    x = np.concatenate((x, data_df[['center']].values), axis=0)
                else:
                    x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0)
        except FileNotFoundError:
            print("Unable to read file %s" % path)
            continue

    if x is None:
        print("No driving data were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    if cfg.TRACK == "track1":
        print("For %s, we use only the first %d images (~1 lap)" % (cfg.TRACK, cfg.TRACK1_IMG_PER_LAP))
        x = x[:cfg.TRACK1_IMG_PER_LAP]
    else:
        print("Incorrect cfg.TRACK option provided")
        exit()

    try:
        x_train, x_test = train_test_split(x, test_size=cfg.TEST_SIZE, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    duration_train = time.time() - start
    print("Loading training set completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    print("Data set: " + str(len(x)) + " elements")
    print("Training set: " + str(len(x_train)) + " elements")
    print("Test set: " + str(len(x_test)) + " elements")
    return x_train, x_test
