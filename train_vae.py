import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
import utils
from vae_batch_generator import Generator
from config import Config
from variational_autoencoder import VariationalAutoencoder

np.random.seed(0)


def load_data_for_vae(cfg):
    """
    Load training data and split it into training and validation set
    """
    tracks = cfg.TRACK
    drive = utils.get_driving_styles(cfg)

    print("Loading training set " + str(tracks) + str(drive))

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
                        print(
                            "cfg.USE_ONLY_CENTER_IMG = " + str(
                                cfg.USE_ONLY_CENTER_IMG) + ". Loading only front-facing camera images")
                        x = data_df[['center']].values
                    else:
                        print(
                            "cfg.USE_ONLY_CENTER_IMG = " + str(
                                cfg.USE_ONLY_CENTER_IMG) + ". Loading all camera images")
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

    print("Data set: " + str(len(x)) + " elements")
    print("Training set: " + str(len(x_train)) + " elements")
    print("Test set: " + str(len(x_test)) + " elements")
    return x_train, x_test


def train_vae_model(cfg, vae, x_train, x_test):
    """
    Train the VAE model
    """

    start = time.time()

    name = 'sao/temp/' + vae.model_name + '-{epoch:03d}.h5'

    checkpoint = ModelCheckpoint(
        name,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto')

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
                                  callbacks=[checkpoint],
                                  verbose=1)

    duration_train = time.time() - start
    print("Training completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('training VAE (' + str(cfg.NUM_EPOCHS_SAO_MODEL) + ' epochs)')
    plt.legend(['train'], loc='upper left')
    plt.show()
    plt.savefig('history-training-' + str(vae.model_name) + '.png')

    # save the last model (might not be the best)
    model.save("sao/" + str(vae.model_name) + "-final.h5")


def main():
    cfg = Config()
    cfg.from_pyfile("config")

    x_train, x_test = load_data_for_vae(cfg)
    vae = VariationalAutoencoder(model_name="VAE-track1", loss=cfg.LOSS_SAO_MODEL)

    train_vae_model(cfg, vae, x_train, x_test)


if __name__ == '__main__':
    main()
