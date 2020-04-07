import argparse
import base64
import logging
import os
from datetime import datetime

import utils

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np

import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
from utils import rmse

sio = socketio.Server()
app = Flask(__name__)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        # The current speed of the car
        speed = float(data["speed"])

        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        # save frame
        image_path = ''
        if args.data_dir != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.data_dir, args.sim_name, "IMG", timestamp)
            image_path = '{}.jpg'.format(image_filename)
            image.save(image_path)

        try:
            image = np.asarray(image)  # from PIL image to numpy array

            image = utils.preprocess(image)  # apply the preprocessing
            image = np.array([image])  # the model expects 4D array

            # predict the steering angle for the image
            steering_angle = float(model.predict(image, batch_size=1))

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit

            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED

            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

            send_control(steering_angle, throttle)

        except Exception as e:
            print(e)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
        },
        skip_sid=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remote Driving - Data Collection')
    parser.add_argument('-d', help='data save directory', dest='data_dir', type=str,
                        default='simulations')
    parser.add_argument('-n', help='simulation name', dest='sim_name', type=str, default='')
    parser.add_argument('-m', help='path to the model', dest='model', type=str,
                        default="models/model.h5")
    parser.add_argument('-s', help='max speed', dest='max_speed', type=int, default=30)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    if "chauffeur" in args.model:
        model = load_model(args.model, custom_objects={"rmse": rmse})
    else:
        model = load_model(args.model)

    assert model is not None

    MIN_SPEED = 10
    MAX_SPEED = args.max_speed
    speed_limit = MAX_SPEED

    if args.data_dir != '':
        utils.create_output_dir(args, utils.csv_fieldnames_original_simulator)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
