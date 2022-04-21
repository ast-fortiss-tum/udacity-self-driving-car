# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import csv
import gc
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from scipy.stats import gamma
from tqdm import tqdm

import utils
from config import Config
from selforacle.utils_vae import load_vae
from selforacle.vae import normalize_and_reshape, RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS
from utils import load_all_images
from utils import plot_reconstruction_losses
from vae_evaluate_novelty_detection_heatmaps_scores import compute_fp_and_tn, compute_tp_and_fn

np.random.seed(0)


def load_or_compute_losses(anomaly_detector, dataset, cached_file_name, delete_cache):
    losses = []

    current_path = os.getcwd()
    cache_path = os.path.join(current_path, 'cache', cached_file_name + '.npy')

    if delete_cache:
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print("delete_cache = True. Removed losses cache file " + cached_file_name)

    try:
        losses = np.load(cache_path)
        losses = losses.tolist()
        print("Found losses for " + cached_file_name)
        return losses
    except FileNotFoundError:
        print("Losses for " + cached_file_name + " not found. Computing...")

        for x in tqdm(dataset):
            x = utils.resize(x)
            x = normalize_and_reshape(x)

            # sanity check
            # z_mean, z_log_var, z = anomaly_detector.encoder.predict(x)
            # decoded = anomaly_detector.decoder.predict(z)
            # reconstructed = anomaly_detector.predict(x)

            loss = anomaly_detector.test_on_batch(x)[1]  # total loss
            losses.append(loss)

        np_losses = np.array(losses)
        np.save(cache_path, np_losses)
        print("Losses for " + cached_file_name + " saved.")

    return losses


def plot_picture_orig_dec(orig, dec, picture_name, losses, num=10):
    n = num
    plt.figure(figsize=(40, 8))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(orig[i].reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title("Original Photo")

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(dec[i].reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title("Reconstructed loss %.4f" % losses[i])

    plt.savefig(picture_name, bbox_inches='tight')
    plt.show()
    plt.close()


def get_results_mispredictions(cfg, sim_name, name,
                               losses_on_nominal, losses_on_anomalous,
                               data_df_nominal, data_df_anomalous,
                               seconds_to_anticipate):
    false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(data_df_nominal,
                                                                                 losses_on_nominal)

    true_positive_windows, false_negative_windows = compute_tp_and_fn(data_df_anomalous, losses_on_anomalous, threshold,
                                                                      seconds_to_anticipate)

    print("time to misbehaviour (s): %d" % seconds_to_anticipate)

    true_positive_windows = 5 * true_positive_windows
    false_negative_windows = 5 * false_negative_windows

    precision = true_positive_windows / (true_positive_windows + false_positive_windows)
    recall = true_positive_windows / (true_positive_windows + false_negative_windows)
    f1 = (2 * precision * recall) / (precision + recall)

    mcc = (true_positive_windows * true_negative_windows - false_positive_windows * false_negative_windows) / \
          (math.sqrt(
              (true_positive_windows + false_positive_windows) *
              (true_positive_windows + false_negative_windows) *
              (true_negative_windows + false_positive_windows) *
              (true_negative_windows + false_negative_windows)))

    print("Precision: " + str(round(precision * 100)) + "%")
    print("Recall: " + str(round(recall * 100)) + "%")
    print("F-1: " + str(round(f1 * 100)) + "%")
    print("MCC: " + str(round(mcc)))

    # if not os.path.exists('failure_prediction_heatmaps.csv'):
    #     with open('failure_prediction_heatmaps.csv', mode='w', newline='') as class_imbalance_result_file:
    #         writer = csv.writer(class_imbalance_result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
    #                             lineterminator='\n')
    #         writer.writerow(
    #             ["simulation", "autoencoder", "ttd", "precision", "recall", "f1", "auc", "aucprc"])
    #
    #         writer.writerow([sim_name, name, seconds_to_anticipate,
    #                          str(round(precision_score(crashed, prediction) * 100, 1)),
    #                          str(round(recall_score(crashed, prediction) * 100, 1)),
    #                          str(round(f1_score(crashed, prediction) * 100, 1)),
    #                          str(round(roc_auc_score(crashed, prediction), 3)),
    #                          str(round(auc_score, 3))])
    #
    # else:
    #     with open('failure_prediction_heatmaps.csv', mode='a') as novelty_detection_result_file:
    #         writer = csv.writer(novelty_detection_result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
    #                             lineterminator='\n')
    #         writer.writerow([sim_name, name, seconds_to_anticipate,
    #                          str(round(precision_score(crashed, prediction) * 100, 1)),
    #                          str(round(recall_score(crashed, prediction) * 100, 1)),
    #                          str(round(f1_score(crashed, prediction) * 100, 1)),
    #                          str(round(roc_auc_score(crashed, prediction), 3)),
    #                          str(round(auc_score, 3))])
    #         if seconds_to_anticipate == 3:
    #             writer.writerow(["", "", "", "", "", "", "", ""])


def get_threshold(losses, conf_level=0.95):
    # print("Fitting reconstruction error distribution using Gamma distribution")

    # removing zeros
    losses = np.array(losses)
    losses_copy = losses[losses != 0]
    shape, loc, scale = gamma.fit(losses_copy, floc=0)

    # print("Creating thresholds using the confidence intervals: %s" % conf_level)
    t = gamma.ppf(conf_level, shape, loc=loc, scale=scale)
    print('threshold: ' + str(t))
    return t


def get_scores(cfg, name, new_losses, losses, threshold):
    # only occurring when conditions == unexpected
    true_positive = []
    false_negative = []

    # only occurring when conditions == nominal
    false_positive = []
    true_negative = []

    # required for adaptation
    likely_true_positive_unc = []
    likely_false_positive_cte = []
    likely_false_positive_unc = []
    likely_true_positive_cte = []
    likely_true_negative_unc = []
    likely_false_negative_unc = []
    likely_true_negative_cte = []
    likely_false_negative_cte = []

    # get threshold
    if threshold is not None:
        threshold = threshold
    else:
        threshold = get_threshold(losses, conf_level=0.95)

    # load the online uncertainty from csv
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)
    uncertainties = data_df["uncertainty"]
    cte_values = data_df["cte"]
    crashed_values = data_df["crashed"]

    cfg.UNCERTAINTY_TOLERANCE_LEVEL = get_threshold(uncertainties, conf_level=0.95)

    print("loaded %d uncertainty and %d CTE values" % (len(uncertainties), len(cte_values)))

    for idx, loss in enumerate(losses):
        if loss >= threshold:

            # autoencoder based
            if crashed_values[idx] == 0:
                false_positive.append(idx)
            elif crashed_values[idx] == 1:
                true_positive.append(idx)

            # uncertainty based
            if uncertainties[idx] < cfg.UNCERTAINTY_TOLERANCE_LEVEL:
                likely_false_positive_unc.append(idx)
            else:
                likely_true_positive_unc.append(idx)

            # cte based
            if cte_values[idx] < cfg.CTE_TOLERANCE_LEVEL:
                likely_false_positive_cte.append(idx)
            else:
                likely_true_positive_cte.append(idx)

        elif loss < threshold:  # either FN/TN

            # autoencoder based
            if crashed_values[idx] == 0:
                true_negative.append(idx)
            elif crashed_values[idx] == 1:
                false_negative.append(idx)

            # uncertainty based
            if uncertainties[idx] > cfg.UNCERTAINTY_TOLERANCE_LEVEL:
                likely_true_negative_unc.append(idx)
            else:
                likely_false_negative_unc.append(idx)

            # cte based
            if cte_values[idx] > cfg.CTE_TOLERANCE_LEVEL:
                likely_true_negative_cte.append(idx)
            else:
                likely_false_negative_cte.append(idx)

    assert len(losses) == (len(true_positive) + len(false_negative) +
                           len(false_positive) + len(true_negative))

    assert len(losses) == (len(likely_true_positive_unc) + len(likely_false_negative_unc) +
                           len(likely_false_positive_unc) + len(likely_true_negative_unc))

    assert len(losses) == (len(likely_true_positive_cte) + len(likely_false_negative_cte) +
                           len(likely_false_positive_cte) + len(likely_true_negative_cte))

    print("true_positive: %d" % len(true_positive))
    print("false_negative: %d" % len(false_negative))
    print("false_positive: %d" % len(false_positive))
    print("true_negative: %d" % len(true_negative))

    print("")

    print("likely_true_positive (unc): %d" % len(likely_true_positive_unc))
    print("likely_false_negative (unc): %d" % len(likely_false_negative_unc))
    print("likely_false_positive (unc): %d" % len(likely_false_positive_unc))
    print("likely_true_negative (unc): %d" % len(likely_true_negative_unc))

    print("")

    print("likely_true_positive (cte): %d" % len(likely_true_positive_cte))
    print("likely_false_negative (cte): %d" % len(likely_false_negative_cte))
    print("likely_false_positive (cte): %d" % len(likely_false_positive_cte))
    print("likely_true_negative (cte): %d" % len(likely_true_negative_cte))

    # compute average catastrophic forgetting

    catastrophic_forgetting = np.empty(2)
    catastrophic_forgetting[:] = np.NaN
    if losses != new_losses:
        assert len(losses) == len(new_losses)
        errors = list()
        for idx, loss in enumerate(losses):
            loss_original = losses[idx]
            loss_new = new_losses[idx]
            if loss_new > loss_original:
                errors.append(loss_new - loss_original)

        catastrophic_forgetting = list()
        catastrophic_forgetting.append(np.mean(errors))
        catastrophic_forgetting.append(np.std(errors))

        print(
            f"catastrophic forgetting (mean/std): {catastrophic_forgetting[0]:.2f} +- {catastrophic_forgetting[1]:.2f}")

    if not os.path.exists('class_imbalance.csv'):
        with open('class_imbalance.csv', mode='w', newline='') as class_imbalance_result_file:
            writer = csv.writer(class_imbalance_result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(["autoencoder", "fp", "lfp_unc", "lfp_cte", "mean_CF", "std_CF"])
            writer.writerow([name, len(false_positive), len(likely_false_positive_unc), len(likely_false_positive_cte),
                             round(catastrophic_forgetting[0], 4),
                             round(catastrophic_forgetting[1], 4)])
    else:
        with open('class_imbalance.csv', mode='a') as class_imbalance_result_file:
            writer = csv.writer(class_imbalance_result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow([name, len(false_positive), len(likely_false_positive_unc), len(likely_false_positive_cte),
                             round(catastrophic_forgetting[0], 4),
                             round(catastrophic_forgetting[1], 4)])

    return likely_false_positive_unc, likely_false_positive_cte, catastrophic_forgetting


def load_and_eval_vae(cfg, dataset, delete_cache):
    vae, name = load_vae(cfg, load_vae_from_disk=True)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)

    losses = load_or_compute_losses(vae, dataset, name, delete_cache)
    threshold_nominal = get_threshold(losses, conf_level=0.95)
    plot_reconstruction_losses(losses, None, name, threshold_nominal, None, data_df)
    lfp_unc, lfp_cte, _ = get_scores(cfg, name, losses, losses, threshold_nominal)

    del vae
    K.clear_session()
    gc.collect()


def main():
    os.chdir(os.getcwd().replace('script', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    dataset = load_all_images(cfg)
    load_and_eval_vae(cfg, dataset, delete_cache=True)


if __name__ == '__main__':
    main()
