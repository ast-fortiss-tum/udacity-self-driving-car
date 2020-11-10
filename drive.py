import argparse
import base64
import logging
import os
from datetime import datetime
from io import BytesIO
from warnings import simplefilter
import sys
import signal
import subprocess

import eventlet.wsgi
import numpy as np
import socketio
import tensorflow as tf
from PIL import Image
from flask import Flask

import utils
from utils import rmse
from variational_autoencoder import VariationalAutoencoder

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

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

        if int(data["lapNumber"]) > args.max_laps:
            # kill(proc.pid)
            sio.emit('shutdown', data={}, skip_sid=True)  # DO NOT CHANGE THIS

            # os.kill(proc.pid, signal.SIGKILL)
            # exit(0)

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

            image_copy = autoenconder_model.normalize_and_reshape(image_copy)
            loss = anomaly_detection.test_on_batch(image_copy, image_copy)

            image = utils.preprocess(image)  # apply the pre-processing
            image = np.array([image])  # the model expects 4D array

            # predict the steering angle for the given image
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
            elif args.threshold < loss <= args.threshold * 1.1:
                confidence = 0
            else:
                confidence = 1

            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

            global frame_id

            send_control(steering_angle, throttle, confidence, loss, args.max_laps)
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
        sio.emit('manual', data={}, skip_sid=True)  # DO NOT CHANGE THIS


@sio.on('connect')  # DO NOT CHANGE THIS
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0, 1, 0, 1)


@sio.on('disconnect')
def disconnect(sid):
    print("disconnect ", sid)
    sio.disconnect(sid)
    server.stop()
    server.wait()
    sys.exit()


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


def kill(proc_pid):
    import psutil
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remote Driving - Data Collection')
    parser.add_argument('-d', help='data save directory', dest='data_dir', type=str,
                        default='')
    parser.add_argument('-n', help='simulation name', dest='sim_name', type=str, default='dm-track1-35')
    parser.add_argument('-m', help='path to the model', dest='model', type=str,
                        default="model.h5")
    parser.add_argument('-ad', help='path to the anomaly detector model', dest='anomaly_detector', type=str,
                        default="sao/VAE-ICSE20.h5")  # DO NOT CHANGE THIS
    parser.add_argument('-threshold', help='threshold for the outlier detector', dest='threshold', type=float,
                        default=0.035)
    parser.add_argument('-s', help='speed', dest='speed', type=int, default=35)
    parser.add_argument('-max_laps', help='number of laps in a simulation', dest='max_laps', type=int, default=1)

    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    # os.open("/Users/astocco/Desktop/Work/Precrime/simulators/sim-quality-metrics.app/Contents/MacOS/sim-quality-metrics")

    FileName = "/Users/astocco/Desktop/Work/Precrime/simulators/sim-quality-metrics-new.app/Contents/MacOS/sim-quality-metrics-new"
    # proc = subprocess.call(['open', FileName], shell=True)
    proc = subprocess.Popen([FileName, ""], shell=True)

    path = os.path.join(os.getcwd(), 'models', args.model)

    if "chauffeur" in args.model:
        model = tf.keras.models.load_model(path, custom_objects={"rmse": rmse})
    else:
        model = tf.keras.models.load_model(path)

    MAX_SPEED = args.speed
    MIN_SPEED = 10
    speed_limit = MAX_SPEED

    autoenconder_model = VariationalAutoencoder(args.anomaly_detector)
    anomaly_detection = utils.load_autoencoder(autoenconder_model)
    anomaly_detection.compile(optimizer='adam', loss='mean_squared_error')

    if args.data_dir != '':
        utils.create_output_dir(args, utils.csv_fieldnames_improved_simulator)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    server = eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    server.stop()
    server.wait()
