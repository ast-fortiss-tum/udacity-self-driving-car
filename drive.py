import argparse
import base64
import logging, os
from datetime import datetime
import shutil
import csv

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
import utils
from utils import rmse
from variational_autoencoder import VariationalAutoencoder

sio = socketio.Server()
app = Flask(__name__)
model = None

prev_image_array = None
anomaly_detection = None
autoenconder_model = None
frame_id = 0


@sio.on('telemetry')
def telemetry(sid, data):
    if data:

        # The current speed of the car
        speed = float(data["speed"])

        # the current way point and lap
        wayPoint = int(data["currentWayPoint"])
        lapNumber = int(data["lapNumber"])

        # Cross-Track Error: distance from the center of the lane
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
        if args.data_dir != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.data_dir, args.sim_name, "IMG", timestamp)
            image_path = '{}.jpg'.format(image_filename)
            image.save(image_path)

        try:
            image = np.asarray(image)  # from PIL image to numpy array

            image_copy = np.copy(image)
            # image_copy = utils.preprocess(image_copy)  # apply the preprocessing
            image_copy = autoenconder_model.normalize_and_reshape(image_copy)
            loss = anomaly_detection.test_on_batch(image_copy, image_copy)

            image = utils.preprocess(image)  # apply the preprocessing
            image = np.array([image])  # the model expects 4D array
            image = image.astype('float32')

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

            if loss > args.threshold * 1.1:
                confidence = -1
            elif loss > args.threshold and loss <= args.threshold * 1.1:
                confidence = 0
            else:
                confidence = 1

            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

            global frame_id

            send_control(steering_angle, throttle, brake, confidence, loss, args.max_laps)
            if args.data_dir:
                csv_path = os.path.join(args.data_dir, args.sim_name)
                utils.writeCsvLine(csv_path,
                                   [frame_id, args.model, args.anomaly_detector, args.threshold, args.sim_name,
                                    lapNumber, wayPoint, loss, cte, steering_angle, throttle, speed,
                                    brake, isCrash,
                                    distance, sim_time, ang_diff,  # new metrics
                                    image_path, number_obe, number_crashes])

                frame_id = frame_id + 1

        except Exception as e:
            print(e)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0, 0, 1, 0, 1)


def send_control(steering_angle, throttle, brake, confidence, loss, max_laps):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__(),
            'brake': brake.__str__(),
            'confidence': confidence.__str__(),
            'loss': loss.__str__(),
            'max_laps': max_laps.__str__()
        },
        skip_sid=True)


def load_autoencoder(model):
    autoencoder = model._create_keras_model()
    autoencoder.load_weights(args.anomaly_detector_name)
    assert (autoencoder is not None)
    return autoencoder


def writeCsvLine(filename, row):
    if filename is not None:
        filename += "/driving_log.csv"
        with open(filename, mode='a') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(row)
            result_file.flush()
            result_file.close()
    else:
        create_csv_results_file_header(filename)


def create_csv_results_file_header(file_name):
    if file_name is not None:
        file_name += "/driving_log.csv"
        with open(file_name, mode='w', newline='') as result_file:
            csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            fieldnames = ["FrameId", "Self Driving Model", "Anomaly Detector", "Threshold", "Track Name", "Lap Number",
                          "Check Point", "Loss", "Steering Angle", "Throttle", "Brake", "Speed", "Crashed", "center",
                          "Tot OBEs", "Tot Crashes"]
            writer = csv.DictWriter(result_file, fieldnames=fieldnames)
            writer.writeheader()
            result_file.flush()
            result_file.close()

    return None


if __name__ == '__main__':

    with open('C:\\Users\\41763\\repos\\master-thesis-marco-calzana\\self-driving-car\\model_name.txt',
              'r') as modelNameFile:
        fileName = modelNameFile.read()
    modelNameFile.close()

    print(fileName)

    parser = argparse.ArgumentParser(description='Remote Driving - Data Collection')
    parser.add_argument('-d', help='data save directory', dest='data_dir', type=str,
                        default="simulations")
    parser.add_argument('-m', help='path to the model', dest='model', type=str,
                        default="models/epoch-dataset5-304.h5")
                        # default='C:\\Users\\41763\\track1_epochmodels\\'+ fileName)

    parser.add_argument('-ad', help='path to the anomaly detector model', dest='anomaly_detector', type=str,
                        default="sao/VAE-ICSE20.h5")  # DO NOT CHANGE THIS
    parser.add_argument('-threshold', help='threshold for the outlier detector', dest='threshold', type=float,
                        default=0.035)
    parser.add_argument('-n', help='simulation name', dest='sim_name', type=str, default='trial')
    parser.add_argument('-s', help='speed', dest='speed', type=int, default=30)
    parser.add_argument('-max_laps', help='number of laps in a simulation', dest='max_laps', type=int, default=1)

    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    MAX_SPEED = args.speed
    MIN_SPEED = 10
    speed_limit = MAX_SPEED

    if "chauffeur" in args.model:
        model = load_model(args.model, custom_objects={"rmse": rmse})
        MAX_SPEED += 5
        speed_limit = MAX_SPEED
    else:
        model = load_model(args.model)
        MAX_SPEED += 5
        speed_limit = MAX_SPEED

    autoenconder_model = VariationalAutoencoder(args.anomaly_detector)
    anomaly_detection = utils.load_autoencoder(autoenconder_model)
    anomaly_detection.compile(optimizer='adam', loss='mean_squared_error')

    if args.data_dir != '':
        path = os.path.join(args.data_dir, args.sim_name, "IMG")
        csv_path = os.path.join(args.data_dir, args.sim_name)

        if os.path.exists(path):
            print("Deleting existing image folder at {}".format(path))
            shutil.rmtree(csv_path)

        print("Creating image folder at {}".format(path))
        os.makedirs(path)
        create_csv_results_file_header(csv_path)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
