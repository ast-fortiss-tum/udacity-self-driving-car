# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import gc
import os

import numpy as np
import pandas as pd
from keras import backend as K

from config import Config
from vae_evaluate import compute_fp_and_tn, compute_tp_and_fn


def evaluate_novelty_detection(cfg):
    for summary_type in ['-avg', '-avg-grad']:

        print("Using summarization method average" if summary_type is '-avg' else "Using summarization method gradient")

        # 1. load heatmap scores in nominal conditions

        cfg.SIMULATION_NAME = 'gauss-journal-track1-nominal'
        path = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, 'htm-smoothgrad-scores' + summary_type + '.npy')
        original_losses = np.load(path)

        path = os.path.join(cfg.TESTING_DATA_DIR,
                            cfg.SIMULATION_NAME,
                            'heatmaps-smoothgrad',
                            'driving_log.csv')
        data_df_nominal = pd.read_csv(path)
        data_df_nominal['loss'] = original_losses

        # 2. load heatmap scores in anomalous conditions

        # cfg.SIMULATION_NAME = 'gauss-journal-track1-rain'
        cfg.SIMULATION_NAME = 'xai-track1-mutant-first-failure'
        # cfg.SIMULATION_NAME = 'xai-track1-mutant-second-failure'

        path = os.path.join(cfg.TESTING_DATA_DIR, cfg.SIMULATION_NAME, 'htm-smoothgrad-scores' + summary_type + '.npy')
        anomalous_losses = np.load(path)

        path = os.path.join(cfg.TESTING_DATA_DIR,
                            cfg.SIMULATION_NAME,
                            'heatmaps-smoothgrad',
                            'driving_log.csv')
        data_df_anomalous = pd.read_csv(path)
        data_df_anomalous['loss'] = anomalous_losses

        # 3. compute a threshold from nominal conditions, and FP and TN
        false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(data_df_nominal)

        # 4. compute TP and FN using different time to misbehaviour windows
        for seconds in range(1, 7):
            true_positive_windows, false_negative_windows = compute_tp_and_fn(data_df_anomalous,
                                                                              anomalous_losses,
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

    K.clear_session()
    gc.collect()


def main():
    os.chdir(os.getcwd().replace('scripts', ''))
    # print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    evaluate_novelty_detection(cfg)


if __name__ == '__main__':
    main()
