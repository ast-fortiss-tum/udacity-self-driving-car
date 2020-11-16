import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import utils
from config import Config
from vae_batch_generator import Generator
from variational_autoencoder import VariationalAutoencoder

np.random.seed(0)


def load_data_for_vae(cfg):
    """
    Load training data and split it into training and validation set
    """
    tracks = cfg.TRACK
    drive = utils.get_driving_styles(cfg)

    print("Loading training set " + str(tracks) + str(drive))

    start = time.time()

    x = None
    path = None
    x_train = None
    x_test = None

    for track in tracks:
        for drive_style in drive:
            try:
                path = os.path.join(cfg.TRAINING_DATA_DIR, cfg.SIMULATION_DATA_DIR, track, drive_style,
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


def train_vae_model(cfg, vae, name, x_train, x_test):
    """
    Train the VAE model
    """

    my_file = Path(os.path.join(cfg.SAO_MODELS_DIR, name) + '.h5')
    if my_file.exists():
        print("Model %s already exists. Quit training." % str(name))
        return

    start = time.time()

    model = vae.create_autoencoder()

    x_train = shuffle(x_train, random_state=0)
    x_test = shuffle(x_test, random_state=0)
    train_generator = Generator(model, x_train, True, cfg)
    val_generator = Generator(model, x_test, True, cfg)

    history = model.fit_generator(generator=train_generator,
                                  validation_data=val_generator,
                                  shuffle=True,
                                  epochs=cfg.NUM_EPOCHS_SAO_MODEL,
                                  # steps_per_epoch=len(x_train) // cfg.BATCH_SIZE,
                                  verbose=1)

    duration_train = time.time() - start
    print("Training completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('reconstruction loss (' + str(cfg.LOSS_SAO_MODEL) + ')')
    plt.xlabel('epoch')
    plt.title('training VAE (' + str(cfg.NUM_EPOCHS_SAO_MODEL) + ' epochs)')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('history-training-' + str(vae.model_name) + '.png')
    plt.show()

    # save the last model (might not be the best)
    model.save("sao/" + str(vae.model_name) + ".h5")


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

    name = "VAE-" + cfg.TRACK[0] + '-' + cfg.LOSS_SAO_MODEL + 'loss' + use_center + use_crop
    vae = VariationalAutoencoder(model_name=name, loss=cfg.LOSS_SAO_MODEL)

    return vae, name


def run_training(cfg, x_test, x_train):
    vae, name = setup_vae(cfg)
    train_vae_model(cfg, vae, name, x_train, x_test)


def main():
    cfg = Config()
    cfg.from_pyfile("myconfig.py")

    x_train, x_test = load_data_for_vae(cfg)
    run_training(cfg, x_test, x_train)


if __name__ == '__main__':
    main()
