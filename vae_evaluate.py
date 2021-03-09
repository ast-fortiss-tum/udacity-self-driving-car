import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, precision_recall_curve, \
    auc
from tqdm import tqdm

import utils
from config import Config
from utils import load_all_images
from utils import plot_reconstruction_losses
from utils_vae import load_vae
from vae import normalize_and_reshape, RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS

np.random.seed(0)


def load_or_compute_losses(anomaly_detector, dataset, cached_file_name, delete_cache):
    losses = []

    current_path = os.getcwd()
    cache_path = os.path.join(current_path, 'cache', cached_file_name + '.npy')

    if delete_cache:
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print("delete_cache=true. Removed losses cache file " + cached_file_name)

    try:
        losses = np.load(cache_path)
        losses = losses.tolist()
        print("Found losses data_nominal for " + cached_file_name)
        return losses
    except FileNotFoundError:
        print("Losses data_nominal for " + cached_file_name + " not found. Computing...")

        for x in tqdm(dataset):
            x = utils.resize(x)
            x = normalize_and_reshape(x)

            # sanity check
            # z_mean, z_log_var, z = anomaly_detector.encoder.predict(x)
            # decoded = anomaly_detector.decoder.predict(z)
            # reconstructed = anomaly_detector.predict(x)

            # TODO: check the index
            loss = anomaly_detector.test_on_batch(x)[1]  # total loss
            losses.append(loss)

        np_losses = np.array(losses)
        np.save(cache_path, np_losses)
        print("Losses data_nominal for " + cached_file_name + " saved.")

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
    # only occurring when conditions == unexpected
    true_positive_windows = 0
    false_negative_windows = 0

    # only occurring when conditions == nominal
    false_positive_windows = 0
    true_negative_windows = 0

    # get threshold on nominal data_nominal
    threshold = get_threshold(losses_on_nominal, conf_level=0.95)

    ''' 
    prepare dataset to get TP and FN from unexpected
    '''
    number_frames_anomalous = pd.Series.max(data_df_anomalous['frameId'])
    simulation_time_anomalous = pd.Series.max(data_df_anomalous['time'])
    fps_anomalous = number_frames_anomalous // simulation_time_anomalous

    crashed_anomalous = data_df_anomalous['crashed']
    crashed_anomalous.is_copy = None

    # creates the ground truth
    all_first_frame_position_crashed_sequences = []
    for idx, item in enumerate(crashed_anomalous):
        if idx == number_frames_anomalous:
            continue

        if crashed_anomalous[idx] == 0 and crashed_anomalous[idx + 1] == 1:
            first_index_crash = idx + 1
            all_first_frame_position_crashed_sequences.append(first_index_crash)
            print("first_index_crash: %d" % first_index_crash)

    print("identified %d crash sequences" % len(all_first_frame_position_crashed_sequences))
    print(all_first_frame_position_crashed_sequences)

    frames_to_reassign = fps_anomalous * seconds_to_anticipate

    reaction_frames = pd.Series()
    for item in all_first_frame_position_crashed_sequences:
        crashed_anomalous.loc[[item - frames_to_reassign, item]] = 1
        reaction_frames = reaction_frames.append(crashed_anomalous[item - frames_to_reassign:item])

        print("frames between %d and %d have been labelled as 1" % (item - frames_to_reassign, item))
        print("reaction frames size %d" % len(reaction_frames))

    sma_anomalous = pd.Series(losses_on_anomalous)
    # sma_anomalous = losses.rolling(fps_anomalous, min_periods=1).mean()

    # iterate over losses_on_anomalous and crashed_anomalous jointly and remove frames labelled as 0
    assert len(sma_anomalous) == len(crashed_anomalous)
    idx_to_remove = []
    for idx, loss in enumerate(sma_anomalous):
        if crashed_anomalous[idx] == 0:
            idx_to_remove.append(idx)

    crashed_anomalous = crashed_anomalous.drop(crashed_anomalous.index[idx_to_remove])
    sma_anomalous = sma_anomalous.drop(sma_anomalous.index[idx_to_remove])
    num_windows_anomalous = len(crashed_anomalous) // fps_anomalous
    frames_to_remove = (len(crashed_anomalous) - (fps_anomalous * num_windows_anomalous)) - 1
    crashed_anomalous = crashed_anomalous[:-frames_to_remove]
    sma_anomalous = sma_anomalous[:-frames_to_remove]
    assert len(crashed_anomalous) == len(sma_anomalous)

    prediction = []

    for idx, loss in enumerate(sma_anomalous):
        # print("idx %d" % idx)

        if idx != 0 and idx % fps_anomalous == 0:

            # print("window [%d - %d]" % (idx - fps_anomalous, idx))

            window_mean = pd.Series(sma_anomalous.iloc[idx - fps_anomalous:idx]).mean()
            crashed_mean = pd.Series(crashed_anomalous[idx - fps_anomalous:idx]).mean()

            if window_mean >= threshold:
                if crashed_mean > 0:
                    true_positive_windows += 1
                    prediction.extend([1] * fps_anomalous)
                else:
                    raise ValueError

            elif window_mean < threshold:
                if crashed_mean > 0:
                    false_negative_windows += 1
                    prediction.extend([0] * fps_anomalous)
                else:
                    raise ValueError

            # print("true positives: %d - false negatives: %d" % (true_positive_windows, false_negative_windows))

    assert false_negative_windows + true_positive_windows == num_windows_anomalous
    crashed_anomalous = crashed_anomalous[:-1]
    sma_anomalous = sma_anomalous[:-1]

    assert len(prediction) == len(crashed_anomalous) == len(sma_anomalous)

    '''
        prepare dataset to get FP and TN from unexpected
    '''
    number_frames_nominal = pd.Series.max(data_df_nominal['frameId'])
    simulation_time_nominal = pd.Series.max(data_df_nominal['time'])
    fps_nominal = number_frames_nominal // simulation_time_nominal

    crashed_nominal = data_df_nominal['crashed']
    crashed_nominal.is_copy = None

    num_windows_nominal = len(crashed_nominal) // fps_nominal
    num_to_delete = len(crashed_nominal) - (num_windows_nominal * fps_nominal) - 1

    crashed_nominal = crashed_nominal[:-num_to_delete]
    losses_nominal = losses_on_nominal[:-num_to_delete]
    assert len(crashed_nominal) == len(losses_nominal)

    losses = pd.Series(losses_nominal)
    sma_nominal = losses.rolling(fps_nominal, min_periods=1).mean()

    assert len(crashed_nominal) == len(losses) == len(sma_nominal)

    for idx, loss in enumerate(sma_nominal):

        if idx != 0 and idx % fps_nominal == 0:

            # print("window [%d - %d]" % (idx - fps_nominal, idx))

            window_mean = pd.Series(sma_nominal.iloc[idx - fps_nominal:idx]).mean()
            crashed_mean = pd.Series(crashed_nominal[idx - fps_nominal:idx]).mean()

            if window_mean >= threshold:
                if crashed_mean == 0:
                    false_positive_windows += 1
                    prediction.extend([1] * fps_nominal)
                    print("window [%d - %d]" % (idx - fps_nominal, idx))
                else:
                    raise ValueError

            elif window_mean < threshold:
                if crashed_mean == 0:
                    true_negative_windows += 1
                    prediction.extend([0] * fps_nominal)
                    # print("prediction size %d" % len(prediction))
                else:
                    raise ValueError

    print("false positives: %d - true negatives: %d" % (false_positive_windows, true_negative_windows))
    # print("prediction size %d" % len(prediction))
    assert false_positive_windows + true_negative_windows == num_windows_nominal

    crashed_nominal = crashed_nominal[:-1]
    sma_nominal = crashed_nominal[:-1]

    assert len(prediction) == (len(crashed_anomalous) + len(crashed_nominal))
    crashed = pd.concat([crashed_anomalous, crashed_nominal])
    assert len(prediction) == len(crashed)

    print("time to misbehaviour (s): %d" % seconds_to_anticipate)

    # Calculate and print precision and recall as percentages
    print("Precision: " + str(round(precision_score(crashed, prediction) * 100, 1)) + " % ")
    print("Recall: " + str(round(recall_score(crashed, prediction) * 100, 1)) + " % ")
    # Obtain and print F1 score as a percentage
    print("F1 score: " + str(round(f1_score(crashed, prediction) * 100, 1)) + " %")

    fpr, tpr, thresholds = roc_curve(crashed, prediction)
    # Obtain and print AUC-ROC
    print("AUC-ROC: " + str(round(roc_auc_score(crashed, prediction), 3)))

    # plt.figure()
    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % round(roc_auc_score(crashed, prediction), 3))
    # plt.plot([0, 1], [0, 1], color='black', label="Random", linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic (detection time = %d s)' % seconds_to_anticipate)
    # plt.legend(loc="lower right")
    # plt.show()

    precision, recall, _ = precision_recall_curve(crashed, prediction)
    auc_score = auc(recall, precision)
    print("AUC-PRC: " + str(round(auc_score, 3)) + "\n")

    # plt.figure()
    # plt.plot(recall, precision, label='PR curve (area = %0.2f)' % round(auc_score, 3))
    # plt.plot([0, 1], [1, 0], color='black', label="Random", linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('PR Curve (detection time = %d s)' % seconds_to_anticipate)
    # plt.legend(loc="lower right")
    # plt.show()

    if not os.path.exists('novelty_detection.csv'):
        with open('novelty_detection.csv', mode='w', newline='') as class_imbalance_result_file:
            writer = csv.writer(class_imbalance_result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(
                ["simulation", "autoencoder", "ttd", "precision", "recall", "f1", "auc", "aucprc"])

            writer.writerow([sim_name, name, seconds_to_anticipate,
                             str(round(precision_score(crashed, prediction) * 100, 1)),
                             str(round(recall_score(crashed, prediction) * 100, 1)),
                             str(round(f1_score(crashed, prediction) * 100, 1)),
                             str(round(roc_auc_score(crashed, prediction), 3)),
                             str(round(auc_score, 3))])

    else:
        with open('novelty_detection.csv', mode='a') as novelty_detection_result_file:
            writer = csv.writer(novelty_detection_result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow([sim_name, name, seconds_to_anticipate,
                             str(round(precision_score(crashed, prediction) * 100, 1)),
                             str(round(recall_score(crashed, prediction) * 100, 1)),
                             str(round(f1_score(crashed, prediction) * 100, 1)),
                             str(round(roc_auc_score(crashed, prediction), 3)),
                             str(round(auc_score, 3))])
            if seconds_to_anticipate == 3:
                writer.writerow(["", "", "", "", "", "", "", ""])


def get_threshold(losses, conf_level=0.95):
    # print("Fitting reconstruction error distribution using Gamma distribution")

    shape, loc, scale = gamma.fit(losses, floc=0)

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

    losses = load_or_compute_losses(vae, dataset, name, delete_cache)
    threshold_nominal = get_threshold(losses, conf_level=0.95)
    print(threshold_nominal)
    plot_reconstruction_losses(losses, None, name, threshold_nominal, None, None)
    lfp_unc, lfp_cte, _ = get_scores(cfg, name, losses, losses, threshold_nominal)


def main():
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    dataset = load_all_images(cfg)
    load_and_eval_vae(cfg, dataset, delete_cache=True)


if __name__ == '__main__':
    main()
