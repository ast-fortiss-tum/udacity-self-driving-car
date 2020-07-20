import argparse
import logging
import os

import matplotlib.image as mpimg

import utils

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np

from tensorflow.keras.models import load_model
from utils import rmse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remote Driving - Data Collection')
    parser.add_argument('-d', help='data save directory', dest='data_dir', type=str, default='simulations')
    parser.add_argument('-n', help='simulation name', dest='sim_name', type=str, default='track1-sunny')
    parser.add_argument('-m', help='path to the model', dest='model', type=str, default="models/dave2-dataset5-823.h5")
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    # load self-driving car model
    if "chauffeur" in args.model:
        model = load_model(args.model, custom_objects={"rmse": rmse})
    else:
        model = load_model(args.model)

    # load n images from the dataset at sim_name
    data_df = utils.load_driving_data(args)
    data_df = data_df[['steering_angle', 'center']]
    data_df = data_df[50:55]

    for index, row in data_df.iterrows():
        image = mpimg.imread(row["center"])
        image = np.asarray(image)  # from PIL image to numpy array
        image = utils.preprocess(image)  # apply the pre-processing
        image = np.array([image])  # the model expects 4D array

        # predict the steering angle for the image
        list = []
        for i in range(10):
            steering_angle = float(model.predict(image, batch_size=1))
            list.append(steering_angle)
        print(row["steering_angle"], np.mean(list))
