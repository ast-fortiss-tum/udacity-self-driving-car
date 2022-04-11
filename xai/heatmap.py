# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
from pathlib import Path

from tf_keras_vis.saliency import Saliency
from tqdm import tqdm

import utils
from utils import *


def score_when_decrease(output):
    return -1.0 * output[:, 0]


def compute_heatmap(simulation_name=None, attention_type="SmoothGrad"):
    if simulation_name is None:
        simulation_name = cfg.SIMULATION_NAME

    print("Computing attention heatmaps for simulation %s using %s" % (simulation_name, attention_type))

    # load the image file paths from csv
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        simulation_name,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    data = data_df["center"]
    print("read %d images from file" % len(data))
    # load self-driving car model
    self_driving_car_model = tensorflow.keras.models.load_model(
        Path(os.path.join(cfg.SDC_MODELS_DIR, cfg.SDC_MODEL_NAME)))
    # load attention model
    saliency = None
    if attention_type == "SmoothGrad":
        saliency = Saliency(self_driving_car_model, model_modifier=None)
    avg_heatmaps = []
    avg_gradient_heatmaps = []
    list_of_image_paths = []
    total_time = 0
    prev_hm = gradient = np.zeros((80, 160))
    # create directory for the heatmaps
    path_save_heatmaps = os.path.join(cfg.TESTING_DATA_DIR,
                                      cfg.SIMULATION_NAME,
                                      "heatmaps-" + attention_type.lower(),
                                      "IMG")
    if os.path.exists(path_save_heatmaps):
        print("Deleting folder at {}".format(path_save_heatmaps))
        shutil.rmtree(path_save_heatmaps)
    print("Creating image folder at {}".format(path_save_heatmaps))
    os.makedirs(path_save_heatmaps)
    for idx, img in enumerate(tqdm(data)):

        # convert Windows path, if needed
        if "\\" in img:
            img = img.replace("\\", "/")

        # load image
        x = mpimg.imread(img)

        # preprocess image
        x = utils.resize(x).astype('float32')

        # compute heatmap image
        saliency_map = None
        if attention_type == "SmoothGrad":
            saliency_map = saliency(score_when_decrease, x, smooth_samples=20, smooth_noise=0.20)

        # compute average of the heatmap
        average = np.average(saliency_map)

        # compute gradient of the heatmap
        if idx == 0:
            gradient = 0
        else:
            gradient = abs(prev_hm - saliency_map)
        average_gradient = np.average(gradient)
        prev_hm = saliency_map

        # store the heatmaps
        file_name = img.split('/')[-1]
        file_name = "htm-" + attention_type.lower() + '-' + file_name
        path_name = os.path.join(path_save_heatmaps, file_name)
        mpimg.imsave(path_name, np.squeeze(saliency_map))

        list_of_image_paths.append(path_name)

        avg_heatmaps.append(average)
        avg_gradient_heatmaps.append(average_gradient)
    # save scores
    file_name = "htm-" + attention_type.lower() + '-scores'
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             cfg.SIMULATION_NAME,
                             file_name + '-avg')
    np.save(path_name, avg_heatmaps)
    path_name = os.path.join(cfg.TESTING_DATA_DIR,
                             cfg.SIMULATION_NAME,
                             file_name + '-avg-grad')
    np.save(path_name, avg_heatmaps)
    # save as csv
    df = pd.DataFrame(list_of_image_paths, columns=['center'])
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    data = data_df[["frameId", "time", "crashed"]]
    # copy frame id, simulation time and crashed information from simulation's csv
    df['frameId'] = data['frameId'].copy()
    df['time'] = data['time'].copy()
    df['crashed'] = data['crashed'].copy()
    # save it as a separate csv
    df.to_csv(os.path.join(cfg.TESTING_DATA_DIR,
                           cfg.SIMULATION_NAME,
                           "heatmaps-" + attention_type.lower(),
                           'driving_log.csv'), index=False)


if __name__ == '__main__':
    '''
    Given a simulation  by Udacity, the script reads the corresponding image paths from the csv and creates a heatmap for
    each driving image. The heatmap is created with the SmoothGrad algorithm available from tf-keras-vis 
    (https://keisen.github.io/tf-keras-vis-docs/examples/attentions.html#SmoothGrad). The scripts generates a separate 
    IMG/ folder and csv file.
    '''

    os.chdir(os.getcwd().replace('xai', ''))

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    attention = "SmoothGrad"

    compute_heatmap(simulation_name=None, attention_type=attention)
