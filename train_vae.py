import datetime
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import shutil

import utils
from config import Config
from utils import plot_history
from vae_batch_generator import Generator
from variational_autoencoder import VariationalAutoencoder

np.random.seed(0)

from tensorflow.python.framework import tensor_util
from tensorflow.python import keras


def is_tensor(x):
    return tensor_util.is_tensor(x)


def load_data_for_vae(cfg):
    """
    Load training data and split it into training and validation set
    """
    drive = utils.get_driving_styles(cfg)

    print("Loading training set " + str(cfg.TRACK) + str(drive))

    start = time.time()

    x = None
    path = None
    x_train = None
    x_test = None

    for drive_style in drive:
        try:
            path = os.path.join(cfg.TRAINING_DATA_DIR,
                                cfg.SIMULATION_DATA_DIR,
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


def train_vae_model(cfg, vae, name, x_train, x_test, delete_model):
    """
    Train the VAE model
    """

    # TODO: do not use .h5 extension when saving/loading custom objects. No longer compatible across platforms!
    my_file = Path(os.path.join(cfg.SAO_MODELS_DIR, name))

    if delete_model:
        print("Deleting model %s" % str(my_file))
        shutil.rmtree(my_file, ignore_errors=True)
        print("Model %s deleted" % str(my_file))

    if my_file.exists():
        print("Model %s already exists. Quit training." % str(name))
        return

    es = keras.callbacks.EarlyStopping(monitor='loss', patience=5, mode="auto", restore_best_weights=True)

    start = time.time()

    model, encoder, decoder = vae.create_autoencoder()

    x_train = shuffle(x_train, random_state=0)
    x_test = shuffle(x_test, random_state=0)
    train_generator = Generator(model, x_train, True, cfg)
    val_generator = Generator(model, x_test, True, cfg)

    history = model.fit(train_generator,
                        validation_data=val_generator,
                        shuffle=False,
                        epochs=cfg.NUM_EPOCHS_SAO_MODEL,
                        callbacks=[es],
                        # steps_per_epoch=len(x_train) // cfg.BATCH_SIZE,
                        verbose=1)

    duration_train = time.time() - start
    print("Training completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    plot_history(history.history, cfg, name, vae)

    # save the last model (might not be the best)
    # TODO: save the encoder and decoder as well
    # TODO: do not use .h5 extension when saving/loading custom objects. No longer compatible across platforms!
    tensorflow.keras.models.save_model(model, my_file.__str__())

    encoder_name = name.replace("VAE-", "encoder-")
    encoder_file = Path(os.path.join(cfg.SAO_MODELS_DIR, encoder_name))
    tensorflow.keras.models.save_model(encoder, encoder_file.__str__())

    decoder_name = name.replace("VAE-", "decoder-")
    decoder_file = Path(os.path.join(cfg.SAO_MODELS_DIR, decoder_name))
    tensorflow.keras.models.save_model(decoder, decoder_file.__str__())

    # save history file
    np.save(Path(os.path.join(cfg.SAO_MODELS_DIR, name)).__str__() + ".npy", history.history)


def setup_vae(cfg):
    if cfg.USE_ONLY_CENTER_IMG:
        print("cfg.USE_ONLY_CENTER_IMG = " + str(cfg.USE_ONLY_CENTER_IMG) + ". Using only front-facing camera images")
        use_center = '-centerimg-'
    else:
        print("cfg.USE_ONLY_CENTER_IMG = " + str(cfg.USE_ONLY_CENTER_IMG) + ". Using all camera images")
        use_center = '-allimg-'

    if cfg.USE_CROP:
        print("cfg.USE_CROP = " + str(cfg.USE_CROP) + ". Cropping the image")
        use_crop = 'usecrop'
    else:
        print("cfg.USE_CROP = " + str(cfg.USE_CROP) + ". Using the entire image")
        use_crop = 'nocrop'

    name = "VAE-" + cfg.TRACK + '-' + cfg.LOSS_SAO_MODEL + 'loss' + use_center + use_crop
    vae = VariationalAutoencoder(model_name=name, loss=cfg.LOSS_SAO_MODEL)

    return vae, name


def run_training(cfg, x_test, x_train):
    vae, name = setup_vae(cfg)
    train_vae_model(cfg, vae, name, x_train, x_test, delete_model=True)


def main():
    cfg = Config()
    cfg.from_pyfile("myconfig.py")

    x_train, x_test = load_data_for_vae(cfg)
    run_training(cfg, x_test, x_train)


if __name__ == '__main__':
    main()
