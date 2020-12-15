import base64
import logging
import os
from datetime import datetime
from pathlib import Path

import tensorflow
from tensorflow import keras

import utils
from config import Config

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

from tensorflow.keras.models import load_model
from utils import rmse, crop, resize
from vae import VAE, normalize_and_reshape

sio = socketio.Server()
app = Flask(__name__)
model = None

prev_image_array = None
anomaly_detection = None
autoenconder_model = None
frame_id = 0
batch_size = 4
sampling = 15
uncertainty = -1


@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        # The current speed of the car
        speed = float(data["speed"])

        # the current way point and lap
        wayPoint = int(data["currentWayPoint"])
        lapNumber = int(data["lapNumber"])

        # Cross-Track Error
        cte = float(data["cte"])

        # brake
        brake = float(data["brake"])
        # print("brake: %.2f" % brake)

        # the distance driven by the car
        distance = float(data["distance"])

        # the time driven by the car
        sim_time = int(data["sim_time"])
        # print(sim_time)

        # the angular difference
        ang_diff = float(data["ang_diff"])

        # whether an OBE or crash occurred
        isCrash = int(data["crash"])

        # the total number of OBEs and crashes so far
        number_obe = int(data["tot_obes"])
        number_crashes = int(data["tot_crashes"])

        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        # save frame
        image_path = ''
        if cfg.TESTING_DATA_DIR != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, "IMG", timestamp)
            image_path = '{}.jpg'.format(image_filename)
            image.save(image_path)

        try:
            image = np.asarray(image)  # from PIL image to numpy array
            image_copy = np.copy(image)

            if cfg.USE_CROP:
                image_copy = crop(image_copy)

            image_copy = resize(image_copy)
            image_copy = normalize_and_reshape(image_copy)
            loss = anomaly_detection.test_on_batch(image_copy, image_copy)[1]

            image = utils.preprocess(image)  # apply the pre-processing
            image = np.array([image])  # the model expects 4D array

            global steering_angle
            global uncertainty
            global batch_size
            global sampling

            # predict the steering angle for the image
            if cfg.USE_PREDICTIVE_UNCERTAINTY:

                # take batch of data
                x = [image for i in range(batch_size)]

                # init empty predictions
                y_ = np.zeros(batch_size)

                for sample_id in range(batch_size):
                    # save predictions from a sample pass
                    y_[sample_id] = model.predict(x, batch_size)

                # average over all passes if the final steering angle
                steering_angle = y_.mean(axis=0)
                # evaluate against labels
                uncertainty = y_.var(axis=0)
            else:
                steering_angle = float(model.predict(image, batch_size=1))

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit

            if speed > speed_limit:
                speed_limit = cfg.MIN_SPEED  # slow down
            else:
                speed_limit = cfg.MAX_SPEED

            if loss > cfg.SAO_THRESHOLD * 1.1:
                confidence = -1
            elif cfg.SAO_THRESHOLD < loss <= cfg.SAO_THRESHOLD * 1.1:
                confidence = 0
            else:
                confidence = 1

            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

            global frame_id

            send_control(steering_angle, throttle, confidence, loss, cfg.MAX_LAPS)
            if cfg.TESTING_DATA_DIR:
                csv_path = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME)
                utils.write_csv_line(csv_path,
                                     [frame_id, cfg.SDC_MODEL_NAME, cfg.ANOMALY_DETECTOR_NAME, cfg.SAO_THRESHOLD,
                                      cfg.SIMULATION_NAME, lapNumber, wayPoint, loss,
                                      uncertainty,  # new metrics
                                      cte, steering_angle, throttle, speed, brake, isCrash,
                                      distance, sim_time, ang_diff,  # new metrics
                                      image_path, number_obe, number_crashes])

                frame_id = frame_id + 1

        except Exception as e:
            print(e)

    else:
        sio.emit('manual', data={}, skip_sid=True)  # DO NOT CHANGE THIS


@sio.on('connect')  # DO NOT CHANGE THIS
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0, 1, 0, 1)


def send_control(steering_angle, throttle, confidence, loss, max_laps):  # DO NOT CHANGE THIS
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
            'confidence': confidence.__str__(),
            'loss': loss.__str__(),
            'max_laps': max_laps.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    # load the self-driving car model
    model_path = Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME))
    if "chauffeur" in cfg.SDC_MODEL_NAME:
        model = load_model(model_path, custom_objects={"rmse": rmse})
    elif "dave2" in cfg.SDC_MODEL_NAME or "epoch" in cfg.SDC_MODEL_NAME or "commaai" in cfg.SDC_MODEL_NAME:
        model = load_model(model_path)
    else:
        print("cfg.SDC_MODEL_NAME option unknown. Exiting...")
        exit()

    # load the self-assessment oracle model
    encoder = tensorflow.keras.models.load_model('sao/' + 'encoder-' + cfg.ANOMALY_DETECTOR_NAME)
    decoder = tensorflow.keras.models.load_model('sao/' + 'decoder-' + cfg.ANOMALY_DETECTOR_NAME)

    anomaly_detection = VAE(model_name=cfg.ANOMALY_DETECTOR_NAME,
                            loss=cfg.LOSS_SAO_MODEL,
                            encoder=encoder,
                            decoder=decoder)
    anomaly_detection.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    # create the output dir
    if cfg.TESTING_DATA_DIR != '':
        utils.create_output_dir(cfg, utils.csv_fieldnames_improved_simulator)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
