# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import csv
import gc
import os

import pandas as pd
from keras import backend as K

from config import Config
from selforacle import utils_vae
from utils import load_all_heatmaps
from vae_evaluate import load_or_compute_losses
from evaluate_failure_prediction_heatmaps_scores import compute_fp_and_tn, compute_tp_and_fn


def evaluate_failure_prediction(cfg, heatmap_type, simulation_name, aggregation_method, condition, num_samples):
    # 1. compute the nominal threshold
    cfg.SIMULATION_NAME = 'gauss-journal-track1-nominal'
    dataset = load_all_heatmaps(cfg)

    if num_samples is not "all":
        dataset = dataset[:num_samples]  # we limit the number of samples to avoid unbalance

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'heatmaps-' + heatmap_type,
                        'driving_log.csv')
    data_df_nominal = pd.read_csv(path)

    if num_samples is not "all":
        data_df_nominal = data_df_nominal[:num_samples]  # we limit the number of samples to avoid unbalance

    name_of_autoencoder = "track1-VAE-latent16-heatmaps-" + heatmap_type

    vae = utils_vae.load_vae_by_name(name_of_autoencoder)

    name_of_losses_file = "track1-VAE-latent16-heatmaps-" \
                          + heatmap_type + '-' \
                          + cfg.SIMULATION_NAME.replace("/", "-") + "-" \
                          + str(num_samples)
    original_losses = load_or_compute_losses(vae, dataset, name_of_losses_file, delete_cache=False)
    data_df_nominal['loss'] = original_losses

    # 2. evaluate on anomalous conditions
    cfg.SIMULATION_NAME = simulation_name

    dataset = load_all_heatmaps(cfg)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'heatmaps-' + heatmap_type,
                        'driving_log.csv')
    data_df_anomalous = pd.read_csv(path)

    name_of_losses_file = "track1-VAE-latent16-heatmaps-" + heatmap_type + '-' + cfg.SIMULATION_NAME.replace("/",
                                                                                                             "-").replace(
        "\\", "-")

    new_losses = load_or_compute_losses(vae, dataset, name_of_losses_file, delete_cache=False)
    data_df_anomalous['loss'] = new_losses

    false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(data_df_nominal, aggregation_method)

    for seconds in range(1, 4):
        true_positive_windows, false_negative_windows, undetectable_windows = compute_tp_and_fn(data_df_anomalous,
                                                                                                new_losses,
                                                                                                threshold,
                                                                                                seconds,
                                                                                                aggregation_method)

        if true_positive_windows != 0:
            precision = true_positive_windows / (true_positive_windows + false_positive_windows)
            recall = true_positive_windows / (true_positive_windows + false_negative_windows)
            accuracy = (true_positive_windows + true_negative_windows) / (
                    true_positive_windows + true_negative_windows + false_positive_windows + false_negative_windows)

            if precision != 0 or recall != 0:
                f1 = (2 * precision * recall) / (precision + recall)

                print("Precision: " + str(round(precision * 100)) + "%")
                print("Accuracy: " + str(round(accuracy * 100)) + "%")
                print("Recall: " + str(round(recall * 100)) + "%")
                print("F-1: " + str(round(f1 * 100)) + "%\n")
            else:
                precision = recall = f1 = accuracy = 0
                print("Precision: undefined")
                print("Accuracy: undefined")
                print("Recall: undefined")
                print("F-1: undefined\n")
        else:
            precision = recall = f1 = accuracy = 0
            print("Precision: undefined")
            print("Accuracy: undefined")
            print("Recall: undefined")
            print("F-1: undefined\n")

        # 5. write results in a CSV files
        if not os.path.exists("track1-VAE-latent16-heatmaps-" + heatmap_type + '-' + str(condition) + '-' + str(
                num_samples) + '.csv'):
            with open("track1-VAE-latent16-heatmaps-" + heatmap_type + '-' + str(condition) + '-' + str(
                    num_samples) + '.csv', mode='w', newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow(
                    ["heatmap_type", "summarization_method", "aggregation_type", "simulation_name", "failures",
                     "detected", "undetected", "undetectable", "ttm", "precision", 'accuracy', "recall", "f1"])
                writer.writerow(["track1-VAE-latent16-heatmaps-" + heatmap_type, 'rec. loss', aggregation_method,
                                 simulation_name,
                                 str(true_positive_windows + false_negative_windows),
                                 str(true_positive_windows),
                                 str(false_negative_windows),
                                 str(undetectable_windows),
                                 seconds,
                                 str(round(precision * 100)),
                                 str(round(accuracy * 100)),
                                 str(round(recall * 100)),
                                 str(round(f1 * 100))])

        else:
            with open("track1-VAE-latent16-heatmaps-" + heatmap_type + '-' + str(condition) + '-' + str(
                    num_samples) + '.csv', mode='a', newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow(["track1-VAE-latent16-heatmaps-" + heatmap_type, 'rec. loss', aggregation_method,
                                 simulation_name,
                                 str(true_positive_windows + false_negative_windows),
                                 str(true_positive_windows),
                                 str(false_negative_windows),
                                 str(undetectable_windows),
                                 seconds,
                                 str(round(precision * 100)),
                                 str(round(accuracy * 100)),
                                 str(round(recall * 100)),
                                 str(round(f1 * 100))])

    del vae
    K.clear_session()
    gc.collect()


def main():
    os.chdir(os.getcwd().replace('scripts', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    cfg.SIMULATION_NAME = 'xai-track1-fog-10'

    evaluate_failure_prediction(cfg,
                                heatmap_type='smoothgrad',
                                simulation_name=cfg.SIMULATION_NAME,
                                aggregation_method='max')


if __name__ == '__main__':
    main()
