import csv
import datetime
import os
import shutil
import time

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K

RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH = 80, 160
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
# INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
INPUT_SHAPE = (RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS)

csv_fieldnames_original_simulator = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
csv_fieldnames_improved_simulator = ["frameId", "model", "anomaly_detector", "threshold", "sim_name",
                                     "lap", "waypoint", "loss",
                                     "uncertainty",  # newly added
                                     "cte", "steering_angle", "throttle", "speed", "brake", "crashed",
                                     "distance", "time", "ang_diff",  # newly added
                                     "center", "tot_OBEs", "tot_crashes"]


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    image_dir = 'datasets/' + data_dir
    local_path = "/".join(image_file.split("/")[-4:-1]) + "/" + image_file.split("/")[-1]
    img_path = "{0}/{1}".format(image_dir, local_path)
    try:
        return mpimg.imread(img_path)
    except FileNotFoundError:
        print(image_file + " not found")
        # exit(1)


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :]  # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    # return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return cv2.cvtColor(image.astype('uint8') * 255, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image vertically and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augmented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    # TODO: flip should be applied to left/right only and w/ no probability
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def rmse(y_true, y_pred):
    """
    Calculates RMSE
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def write_csv_line(filename, row):
    if filename is not None:
        filename += "/driving_log.csv"
        with open(filename, mode='a') as result_file:
            writer = csv.writer(result_file,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(row)
            result_file.flush()
            result_file.close()
    else:
        create_csv_results_file_header(filename)


def create_csv_results_file_header(file_name, fieldnames):
    if file_name is not None:
        file_name += "/driving_log.csv"
        with open(file_name, mode='w', newline='') as result_file:
            csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer = csv.DictWriter(result_file, fieldnames=fieldnames)
            writer.writeheader()
            result_file.flush()
            result_file.close()

    return None


def create_output_dir(cfg, fieldnames):
    path = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, "IMG")
    csv_path = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME)

    if os.path.exists(path):
        print("Deleting image folder at {}".format(path))
        shutil.rmtree(csv_path)

    print("Creating image folder at {}".format(path))
    os.makedirs(path)
    create_csv_results_file_header(csv_path, fieldnames)


def load_driving_data(cfg: object) -> object:
    path = None
    data_df = None
    try:
        path = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, 'driving_log.csv')
        data_df = pd.read_csv(path, keep_default_na=False)
    except FileNotFoundError:
        print("Unable to read file %s" % path)
        exit()

    return data_df


def get_driving_styles(cfg):
    if cfg.TRACK == "track1":
        return cfg.TRACK1_DRIVING_STYLES
    elif cfg.TRACK == "track2":
        return cfg.TRACK2_DRIVING_STYLES
    elif cfg.TRACK == "track3":
        return cfg.TRACK3_DRIVING_STYLES
    else:
        print("Invalid TRACK option within the config file")
        exit(1)


def plot_history(history, cfg, vae):
    plt.plot(history['loss'])
    plt.plot(history['val_total_loss'])
    plt.ylabel('reconstruction loss (' + str(cfg.LOSS_SAO_MODEL) + ')')
    plt.xlabel('epoch')
    plt.title('training-' + str(vae.model_name))
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/history-training-' + str(vae.model_name) + '.png')

    plt.show()


def load_improvement_set(cfg, ids):
    start = time.time()

    x = None
    path = None

    try:
        path = os.path.join(cfg.TESTING_DATA_DIR,
                            cfg.SIMULATION_NAME,
                            'driving_log.csv')
        data_df = pd.read_csv(path)

        print("Filtering only false positives")
        data_df = data_df[data_df['frameId'].isin(ids)]

        if x is None:
            x = data_df[['center']].values
        else:
            x = np.concatenate((x, data_df[['center']].values), axis=0)

    except FileNotFoundError:
        print("Unable to read file %s" % path)

    if x is None:
        print("No driving data were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    # x = x[:6]

    duration_train = time.time() - start
    print("Loading improvement data set completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    print("False positive data set: " + str(len(x)) + " elements")

    return x


def load_all_images(cfg):
    """
    Load all driving images
    TODO: inefficient
    """
    cfg.TRACK1_DRIVING_STYLES = ["normal"]
    drive = get_driving_styles(cfg)

    print("Loading data set " + str(cfg.TRACK) + str(drive))

    start = time.time()

    x = None
    path = None

    for drive_style in drive:
        try:
            path = os.path.join(cfg.TRAINING_DATA_DIR,
                                cfg.TRAINING_SET_DIR,
                                cfg.TRACK,
                                drive_style,
                                'driving_log.csv')
            data_df = pd.read_csv(path)

            if x is None:
                x = data_df[['center']].values
            else:
                x = np.concatenate((x, data_df[['center']].values), axis=0)

            if cfg.TRACK == "track1":
                # print("Loading only the first 1102 images from %s (one lap)" % track)
                x = x[:cfg.TRACK1_IMG_PER_LAP]
            else:
                print("Not yet implemented! Quitting...")
                exit()

        except FileNotFoundError:
            print("Unable to read file %s" % path)
            continue

    if x is None:
        print("No driving data were provided for training. Provide correct paths to the driving_log.csv files")
        exit()

    # load the images
    images = np.empty([len(x), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

    # TODO: loading only center image for now.
    print("WARNING! For evaluation, loading only front-facing images")

    for i, path in enumerate(x):
        image = load_image(cfg.TRAINING_SET_DIR, path[0])  # load center images

        # visualize whether the input image as expected
        # import matplotlib.pyplot as plt
        # plt.imshow(image)
        # plt.show()

        images[i] = image

    duration_train = time.time() - start
    print("Loading data set completed in %s." % str(datetime.timedelta(seconds=round(duration_train))))

    print("Data set: " + str(len(images)) + " elements")

    return images


def plot_reconstruction_losses(losses, name, threshold):
    # plt.hist(losses, bins=len(losses) // 5)  # TODO: find an appropriate constant
    # plt.show()
    # plt.clf()

    plt.figure(figsize=(20, 4))
    x_losses = np.arange(len(losses))

    x_threshold = np.arange(len(x_losses))
    y_threshold = [threshold] * len(x_threshold)
    plt.plot(x_threshold, y_threshold, color='red', alpha=0.2)

    plt.plot(x_losses, losses, color='blue', alpha=0.7)

    plt.ylabel('Loss')
    plt.xlabel('Number of Instances')
    plt.title("Reconstruction error for " + name)

    plt.savefig('plots/reconstruction-plot-' + name + '.png')

    plt.show()


def laplacian_variance(images):
    return [cv2.Laplacian(image, cv2.CV_32F).var() for image in images]
