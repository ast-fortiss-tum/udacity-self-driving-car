import os

import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow import keras

from config import Config
from utils import load_all_images
from utils import plot_reconstruction_losses, load_improvement_set
from utils_vae import load_vae, load_data_for_vae_retraining
from vae import VAE
from vae_evaluate import load_or_compute_losses, get_threshold, get_scores
from vae_train import train_vae_model


def evaluate_class_imbalance(cfg):
    # remove old files
    if os.path.exists('lfp_unc_before.npy'):
        os.remove('lfp_unc_before.npy')
    if os.path.exists('lfp_cte_before.np'):
        os.remove('lfp_cte_before.npy')

    # 1. compute reconstruction error on nominal images
    dataset = load_all_images(cfg)
    vae, name = load_vae(cfg, load_vae_from_disk=True)
    losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)
    threshold_nominal = get_threshold(losses, conf_level=0.95)
    # plot_reconstruction_losses(losses, None, name, threshold_nominal, None)
    lfp_unc, lfp_cte = get_scores(cfg, name, losses, threshold_nominal)

    np.save('lfp_unc_before.npy', lfp_unc)
    np.save('lfp_cte_before.npy', lfp_cte)

    # exit()

    # 2. compute improvement set and retrain
    x_train, x_test = load_data_for_vae_retraining(cfg, sampling=1)
    improvement_set = load_improvement_set(cfg, lfp_unc)

    # when using center/left/right images, I have to create 3d arrays
    if cfg.USE_ONLY_CENTER_IMG is False:
        improvement_set_allimg = x_train[:len(improvement_set)]
        for i in range(len(improvement_set)):
            improvement_set_allimg[i][0] = improvement_set[i][0]
            improvement_set_allimg[i][1] = improvement_set[i][0]
            improvement_set_allimg[i][2] = improvement_set[i][0]

        improvement_set = improvement_set_allimg

    print("Old training data set: " + str(len(x_train)) + " elements")
    print("Improvement data set: " + str(len(improvement_set)) + " elements")

    initial_improvement_set = improvement_set

    for i in range(cfg.IMPROVEMENT_RATIO - 1):
        temp = initial_improvement_set[:]
        improvement_set = np.concatenate((temp, improvement_set), axis=0)

    x_train_improvement_set, x_test_improvement_set = train_test_split(improvement_set, test_size=cfg.TEST_SIZE,
                                                                       random_state=0)

    # TODO: improvement_set cannot load right/left images and crashes
    x_train = np.concatenate((x_train, x_train_improvement_set), axis=0)
    x_test = np.concatenate((x_test, x_test_improvement_set), axis=0)

    print("New training data set: " + str(len(x_train)) + " elements")

    # magic happens here
    weights = np.array(losses)

    name = name + '-RETRAINED-' + str(cfg.IMPROVEMENT_RATIO) + "X"
    train_vae_model(cfg, vae, name, x_train, x_test, delete_model=True, retraining=True, sample_weights=weights)

    encoder = tensorflow.keras.models.load_model('sao/' + 'encoder-' + name)
    decoder = tensorflow.keras.models.load_model('sao/' + 'decoder-' + name)
    print("loaded adapted VAE from disk")

    vae = VAE(model_name=cfg.ANOMALY_DETECTOR_NAME,
              loss=cfg.LOSS_SAO_MODEL,
              latent_dim=cfg.SAO_LATENT_DIM,
              encoder=encoder,
              decoder=decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE))

    # 3. evaluate w/ old threshold
    new_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)
    # threshold_nominal_new = get_threshold(new_losses, conf_level=0.95)
    plot_reconstruction_losses(losses, new_losses, name, threshold_nominal, None)
    get_scores(cfg, name, new_losses, threshold_nominal)

    # print("Effectiveness on new data")
    # get_scores(cfg, new_losses, threshold_nominal_new)


def main():
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    evaluate_class_imbalance(cfg)


if __name__ == '__main__':
    main()
