import argparse
import os

import numpy as np
import pandas as pd
from keras.layers import Lambda
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from batch_generator import Generator
from utils import INPUT_SHAPE

np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    tracks = ["track1", "track2", "track3"]
    drive = ["normal", "reverse", "sport_normal", "sport_reverse"]

    x = None
    y = None
    for track in tracks:
        for drive_style in drive:
            try:
                path = os.path.join(args.data_dir, track, drive_style, 'driving_log.csv')
                data_df = pd.read_csv(path)
                if x is None:
                    x = data_df[['center', 'left', 'right']].values
                    y = data_df['steering'].values
                else:
                    x = np.concatenate((x, data_df[['center', 'left', 'right']].values), axis=0)
                    y = np.concatenate((y, data_df['steering'].values), axis=0)
            except FileNotFoundError:
                print("Unable to read file %s" % path)
                exit()

    try:
        X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=args.test_size, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    print("Train dataset: " + str(len(X_train)) + " elements")
    print("Test dataset: " + str(len(X_valid)) + " elements")
    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    Modified NVIDIA model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('models/self-driving-car-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate), metrics=['mse'])

    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_valid, y_valid = shuffle(X_valid, y_valid, random_state=0)

    X_train, y_train = X_train[:100], y_train[:100]
    X_valid, y_valid = X_valid[:10], y_valid[:10]

    train_generator = Generator(X_train, y_train, True, args)
    validation_generator = Generator(X_valid, y_valid, False, args)

    model.fit(generator=train_generator,
              validation_data=validation_generator,
              samples_per_epoch=train_generator.nb_samples,
              epochs=args.nb_epoch,
              nb_val_samples=validation_generator.nb_samples,
              use_multiprocessing=False,
              max_queue_size=10,
              workers=4,
              callbacks=[checkpoint],
              verbose=1)

    def s2b(s):
        """
        Converts a string to boolean value
        """
        s = s.lower()
        return s == 'true' or s == 'yes' or s == 'y' or s == '1'

    if __name__ == '__main__':
        """
        Load train/validation data set and train the model
        """
        parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
        parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='datasets/dataset5')
        parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
        parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
        parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=5)
        parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=128)
        parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
        parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
        args = parser.parse_args()

        print('-' * 30)
        print('Parameters')
        print('-' * 30)
        for key, value in vars(args).items():
            print('{:<20} := {}'.format(key, value))
        print('-' * 30)

        data = load_data(args)
        model = build_model(args)
        train_model(model, args, *data)
