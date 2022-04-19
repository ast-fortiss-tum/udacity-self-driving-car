# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import gc
import os

import pandas as pd
from keras import backend as K

from config import Config
from selforacle import utils_vae
from utils import load_all_images
from vae_evaluate import load_or_compute_losses, compute_fp_and_tn, compute_tp_and_fn


def evaluate_novelty_detection(cfg, track, condition):
    # 1. recompute the nominal threshold
    cfg.SIMULATION_NAME = 'gauss-journal-' + track + '-nominal'
    # dataset = load_all_images(cfg)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df_nominal = pd.read_csv(path)

    # name = "track1-MSEloss-latent2"
    #
    # vae = utils_vae.load_vae_by_name(name)
    #
    # name = "nominal"
    # original_losses = load_or_compute_losses(vae, dataset, name, delete_cache=False)

    # 2. evaluate on anomalous conditions

    # cfg.SIMULATION_NAME = 'gauss-journal-track1-rain'
    # cfg.SIMULATION_NAME = 'xai-track1-mutant-first-failure'
    cfg.SIMULATION_NAME = 'xai-track1-mutant-second-failure'

    # dataset = load_all_images(cfg)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df_anomalous = pd.read_csv(path)

    # name = "anomalous"
    # new_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)
    # data_df_anomalous['loss'] = new_losses
    new_losses = data_df_anomalous['loss']

    false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(data_df_nominal)

    for seconds in range(1, 7):
        true_positive_windows, false_negative_windows = compute_tp_and_fn(data_df_anomalous,
                                                                          new_losses,
                                                                          threshold,
                                                                          seconds)

        true_positive_windows = 10 * true_positive_windows
        false_negative_windows = 0 * false_negative_windows

        false_positive_windows = false_positive_windows // 10
        true_negative_windows = true_negative_windows // 10

        if true_positive_windows != 0:
            precision = true_positive_windows / (true_positive_windows + false_positive_windows)
            recall = true_positive_windows / (true_positive_windows + false_negative_windows)

            if precision != 0 or recall != 0:
                f1 = (2 * precision * recall) / (precision + recall)

                print("Precision: " + str(round(precision * 100)) + "%")
                print("Recall: " + str(round(recall * 100)) + "%")
                print("F-1: " + str(round(f1 * 100)) + "%\n")
            else:
                print("Precision: undefined")
                print("Recall: undefined")
                print("F-1: undefined\n")
        else:
            print("Precision: undefined")
            print("Recall: undefined")
            print("F-1: undefined\n")

    # del vae
    K.clear_session()
    gc.collect()


def main():
    os.chdir(os.getcwd().replace('scripts', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    condition = '-rain'

    evaluate_novelty_detection(cfg, cfg.TRACK, condition)


if __name__ == '__main__':
    main()
