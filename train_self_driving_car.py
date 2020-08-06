import argparse
import os

import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

# import logging
#
# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#
# # import warnings filter
# from warnings import simplefilter
#
# # ignore all future warnings
# simplefilter(action='ignore', category=FutureWarning)
from sklearn.utils import shuffle

from batch_generator import Generator
from utils import INPUT_SHAPE
import matplotlib.pyplot as plt

np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    tracks = ["track1", "track2", "track3"]
    drive = ["normal", "reverse", "recovery", "sport_normal", "sport_reverse"]

    x = None
    y = None
    path = None
    x_train = None
    y_train = None
    x_valid = None
    y_valid = None

    for track in tracks:
        for drive_style in drive:
            try:
                path = os.path.join('datasets', args.data_dir, track, drive_style, 'driving_log.csv')
                data_df = pd.read_csv(path)
                if x is None:
                    x = data_df[['center', 'left', 'right']].values
                    y = data_df['steering'].values
                else:
                    x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0)
                    y = np.concatenate((y, data_df['steering'].values), axis=0)
            except FileNotFoundError:
                print("Unable to read file %s" % path)
                continue

    if x is None or y is None:
        print("No driving data were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    try:
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=args.test_size, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    print("Train dataset: " + str(len(x_train)) + " elements")
    print("Test dataset: " + str(len(x_valid)) + " elements")
    return x_train, x_valid, y_train, y_valid


def build_model():
    """
    Modified NVIDIA model
    """
    inputs = keras.Input(shape=INPUT_SHAPE)
    lambda_layer = keras.layers.Lambda(lambda x: x / 127.5 - 1.0, name="lambda_layer")(inputs)
    x = keras.layers.Conv2D(24, (5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(1.0e-6))(lambda_layer)
    x = keras.layers.Dropout(rate=1 - 0.05)(x, training=True)
    x = keras.layers.Conv2D(36, (5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(1.0e-6))(x)
    x = keras.layers.Dropout(rate=1 - 0.05)(x, training=True)
    x = keras.layers.Conv2D(48, (5, 5), activation='relu', strides=(2, 2), kernel_regularizer=l2(1.0e-6))(x)
    x = keras.layers.Dropout(rate=1 - 0.05)(x, training=True)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1.0e-6))(x)
    x = keras.layers.Dropout(rate=1 - 0.05)(x, training=True)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(1.0e-6))(x)
    x = keras.layers.Dropout(rate=1 - 0.05)(x, training=True)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(100, activation='relu', kernel_regularizer=l2(1.0e-6))(x)
    x = keras.layers.Dropout(rate=1 - 0.05)(x, training=True)
    x = keras.layers.Dense(50, activation='relu', kernel_regularizer=l2(1.0e-6))(x)
    x = keras.layers.Dropout(rate=1 - 0.05)(x, training=True)
    x = keras.layers.Dense(10, activation='relu', kernel_regularizer=l2(1.0e-6))(x)
    x = keras.layers.Dropout(rate=1 - 0.05)(x, training=True)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def train_model(model, args, x_train, x_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('self-driving-car-train' + str(args.train_num) + '-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,  # save the model only if the val_loss gets low
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    # shuffle the data because they are sequential; should help over-fitting towards certain parts of the track only
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_valid, y_valid = shuffle(x_valid, y_valid, random_state=0)

    # data for training are augmented, data for validation are not
    train_generator = Generator(x_train, y_train, True, args)
    validation_generator = Generator(x_valid, y_valid, False, args)

    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  epochs=args.nb_epoch,
                                  callbacks=[checkpoint],
                                  verbose=1)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('training' + str(args.train_num))
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    plt.savefig('history-training' + str(args.train_num) + '.png')

    # save the last model anyway (might not be the best)
    model.save("models/model-train" + str(args.train_num) + "-final.h5")


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='dataset5')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=50)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=100)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=64)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    parser.add_argument('-tn', help='training num', dest='train_num', type=int, default=103)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model()

    import time
    start = time.process_time()
    train_model(model, args, *data)
    end = time.process_time()

    print("training finished in %.2f seconds" % (end - start))


if __name__ == '__main__':
    main()
