# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import csv
import gc
import os

import numpy as np
import pandas as pd
from keras import backend as K
from scipy.stats import gamma

from config import Config


def get_threshold(losses, conf_level=0.95):
    print("Fitting reconstruction error distribution using Gamma distribution")

    # removing zeros
    losses = np.array(losses)
    losses_copy = losses[losses != 0]
    shape, loc, scale = gamma.fit(losses_copy, floc=0)

    print("Creating threshold using the confidence intervals: %s" % conf_level)
    t = gamma.ppf(conf_level, shape, loc=loc, scale=scale)
    print('threshold: ' + str(t))
    return t


def evaluate_failure_prediction(cfg, heatmap_type, simulation_name, summary_type, aggregation_method, condition,
                                num_samples):
    print("Using summarization average" if summary_type is '-avg' else "Using summarization gradient")
    print("Using aggregation mean" if aggregation_method is 'mean' else "Using aggregation max")

    # 1. load heatmap scores in nominal conditions

    cfg.SIMULATION_NAME = 'gauss-journal-track1-nominal'
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'htm-' + heatmap_type + '-scores' + summary_type + '.npy')
    original_losses = np.load(path)

    if num_samples is not "all":
        original_losses = original_losses[:num_samples]  # we limit the number of samples to avoid unbalance

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'heatmaps-' + heatmap_type,
                        'driving_log.csv')
    data_df_nominal = pd.read_csv(path)

    if num_samples is not "all":
        data_df_nominal = data_df_nominal[:num_samples]  # we limit the number of samples to avoid unbalance

    data_df_nominal['loss'] = original_losses

    # 2. load heatmap scores in anomalous conditions

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        simulation_name,
                        'htm-' + heatmap_type + '-scores' + summary_type + '.npy')
    anomalous_losses = np.load(path)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        simulation_name,
                        'heatmaps-' + heatmap_type,
                        'driving_log.csv')
    data_df_anomalous = pd.read_csv(path)
    data_df_anomalous['loss'] = anomalous_losses

    # 3. compute a threshold from nominal conditions, and FP and TN
    false_positive_windows, true_negative_windows, threshold = compute_fp_and_tn(data_df_nominal,
                                                                                 aggregation_method)

    # 4. compute TP and FN using different time to misbehaviour windows
    for seconds in range(1, 7):
        true_positive_windows, false_negative_windows, undetectable = compute_tp_and_fn(data_df_anomalous,
                                                                                        anomalous_losses,
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
        if not os.path.exists(heatmap_type + '-' + str(condition) + '-' + str(num_samples) + '.csv'):
            with open(heatmap_type + '-' + str(condition) + '-' + str(num_samples) + '.csv', mode='w',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow(
                    ["heatmap_type", "summarization_method", "aggregation_type", "simulation_name", "failures", "ttm",
                     "precision", 'accuracy', "recall", "f1"])
                writer.writerow([heatmap_type, summary_type[1:], aggregation_method, simulation_name,
                                 str(true_positive_windows + false_negative_windows),
                                 seconds,
                                 str(round(precision * 100)),
                                 str(round(accuracy * 100)),
                                 str(round(recall * 100)),
                                 str(round(f1 * 100))])

        else:
            with open(heatmap_type + '-' + str(condition) + '-' + str(num_samples) + '.csv', mode='a',
                      newline='') as result_file:
                writer = csv.writer(result_file,
                                    delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL,
                                    lineterminator='\n')
                writer.writerow([heatmap_type, summary_type[1:], aggregation_method, simulation_name,
                                 str(true_positive_windows + false_negative_windows),
                                 seconds,
                                 str(round(precision * 100)),
                                 str(round(accuracy * 100)),
                                 str(round(recall * 100)),
                                 str(round(f1 * 100))])

    K.clear_session()
    gc.collect()


def compute_tp_and_fn(data_df_anomalous, losses_on_anomalous, threshold, seconds_to_anticipate,
                      aggregation_method='mean'):
    print("time to misbehaviour (s): %d" % seconds_to_anticipate)

    # only occurring when conditions == unexpected
    true_positive_windows = 0
    false_negative_windows = 0
    undetectable_windows = 0

    ''' 
        prepare dataset to get TP and FN from unexpected
        '''
    number_frames_anomalous = pd.Series.max(data_df_anomalous['frameId'])
    simulation_time_anomalous = pd.Series.max(data_df_anomalous['time'])
    fps_anomalous = number_frames_anomalous // simulation_time_anomalous

    crashed_anomalous = data_df_anomalous['crashed']
    crashed_anomalous.is_copy = None
    crashed_anomalous_in_anomalous_conditions = crashed_anomalous.copy()

    # creates the ground truth
    all_first_frame_position_crashed_sequences = []
    for idx, item in enumerate(crashed_anomalous_in_anomalous_conditions):
        if idx == number_frames_anomalous:
            continue

        if crashed_anomalous_in_anomalous_conditions[idx] == 0 and crashed_anomalous_in_anomalous_conditions[
            idx + 1] == 1:
            first_index_crash = idx + 1
            all_first_frame_position_crashed_sequences.append(first_index_crash)
            # print("first_index_crash: %d" % first_index_crash)

    print("identified %d crash(es)" % len(all_first_frame_position_crashed_sequences))
    print(all_first_frame_position_crashed_sequences)
    frames_to_reassign = fps_anomalous * seconds_to_anticipate  # start of the sequence

    # frames_to_reassign_2 = 1  # first frame before the failure
    frames_to_reassign_2 = fps_anomalous * (seconds_to_anticipate - 1)  # first frame n seconds before the failure

    reaction_window = pd.Series()
    print(all_first_frame_position_crashed_sequences)

    for item in all_first_frame_position_crashed_sequences:
        print("analysing failure %d" % item)

        # the detection window overlaps with a previous crash; skip it
        if crashed_anomalous_in_anomalous_conditions.loc[
           item - frames_to_reassign: item - frames_to_reassign_2].sum() > 2:
            print("failure %d cannot be detected at TTM=%d" % (item, seconds_to_anticipate))
            undetectable_windows += 1
        else:
            crashed_anomalous_in_anomalous_conditions.loc[item - frames_to_reassign: item - frames_to_reassign_2] = 1
            reaction_window = reaction_window.append(
                crashed_anomalous_in_anomalous_conditions[item - frames_to_reassign: item - frames_to_reassign_2])

            print("frames between %d and %d have been labelled as 1" % (
                item - frames_to_reassign, item - frames_to_reassign_2))
            print("reaction frames size is %d" % len(reaction_window))

            sma_anomalous = pd.Series(losses_on_anomalous)
            sma_anomalous = sma_anomalous.iloc[reaction_window.index.to_list()]
            assert len(reaction_window) == len(sma_anomalous)

            # print(sma_anomalous)

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = sma_anomalous.mean()
            elif aggregation_method == "max":
                aggregated_score = sma_anomalous.max()

            print("threshold %s\tmean: %s\tmax: %s" % (
                str(threshold), str(sma_anomalous.mean()), str(sma_anomalous.max())))

            if aggregated_score >= threshold:
                true_positive_windows += 1
            elif aggregated_score < threshold:
                false_negative_windows += 1

        print("failure: %d - true positives: %d - false negatives: %d - undetectable: %d" % (
            item, true_positive_windows, false_negative_windows, undetectable_windows))

    assert len(all_first_frame_position_crashed_sequences) == (
            true_positive_windows + false_negative_windows + undetectable_windows)
    return true_positive_windows, false_negative_windows, undetectable_windows


def compute_fp_and_tn(data_df_nominal, aggregation_method):
    # when conditions == nominal I count only FP and TN
    false_positive_windows = 0
    true_negative_windows = 0

    losses = data_df_nominal['loss'].tolist()

    # get threshold on nominal data
    threshold = get_threshold(losses, conf_level=0.95)

    number_frames_nominal = pd.Series.max(data_df_nominal['frameId'])
    simulation_time_nominal = pd.Series.max(data_df_nominal['time'])
    fps_nominal = number_frames_nominal // simulation_time_nominal

    num_windows_nominal = len(data_df_nominal) // fps_nominal
    if len(data_df_nominal) % fps_nominal != 0:
        num_to_delete = len(data_df_nominal) - (num_windows_nominal * fps_nominal) - 1
        data_df_nominal = data_df_nominal[:-num_to_delete]

    losses = pd.Series(data_df_nominal['loss'])
    sma_nominal = losses.rolling(fps_nominal, min_periods=1).mean()

    for idx, loss in enumerate(sma_nominal):

        if idx > 0 and idx % fps_nominal == 0:

            # print("window [%d - %d]" % (idx - fps_nominal, idx))

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()
            elif aggregation_method == "max":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()

            if aggregated_score >= threshold:
                false_positive_windows += 1
            elif aggregated_score < threshold:
                true_negative_windows += 1

        elif idx == len(sma_nominal) - 1:

            # print("window [%d - %d]" % (idx - fps_nominal, idx))

            aggregated_score = None
            if aggregation_method == "mean":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()
            elif aggregation_method == "max":
                aggregated_score = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).max()

            if aggregated_score >= threshold:
                false_positive_windows += 1
            elif aggregated_score < threshold:
                true_negative_windows += 1

    print("false positives: %d - true negatives: %d" % (false_positive_windows, true_negative_windows))

    assert false_positive_windows + true_negative_windows == num_windows_nominal

    return false_positive_windows, true_negative_windows, threshold


def main():
    os.chdir(os.getcwd().replace('scripts', ''))

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    cfg.SIMULATION_NAME = 'mutants/udacity_change_label_mutated0_MP_25.0_1'

    evaluate_failure_prediction(cfg,
                                heatmap_type='smoothgrad',
                                simulation_name=cfg.SIMULATION_NAME,
                                summary_type='-avg-grad',
                                aggregation_method='max',
                                condition='mutants',
                                num_samples='all')


if __name__ == '__main__':
    main()
