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
    if os.path.exists('likely_false_positive_uncertainty.npy'):
        os.remove('likely_false_positive_uncertainty.npy')
    if os.path.exists('likely_false_positive_cte.npy'):
        os.remove('likely_false_positive_cte.npy')
    if os.path.exists('likely_false_positive_common.npy'):
        os.remove('likely_false_positive_common.npy')

    ''' 
        1. compute reconstruction error on nominal images
        and compute the likely false positives
    '''
    dataset = load_all_images(cfg)
    vae, name = load_vae(cfg, load_vae_from_disk=True)
    original_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)
    threshold_nominal = get_threshold(original_losses, conf_level=0.95)
    likely_fps_uncertainty, likely_fps_cte, _ = get_scores(cfg, name, original_losses, original_losses,
                                                           threshold_nominal)

    assert len(likely_fps_uncertainty) > 0
    assert len(likely_fps_cte) > 0

    likely_fps_common = np.sort(list(set(likely_fps_uncertainty).intersection(likely_fps_cte)))
    assert len(likely_fps_common) > 0

    # save the likely false positive
    np.save('likely_false_positive_uncertainty.npy', likely_fps_uncertainty)
    np.save('likely_false_positive_cte.npy', likely_fps_cte)
    np.save('likely_false_positive_common.npy', likely_fps_common)

    # for mode in ['UNC', 'CTE', 'COM']:
    for mode in ['UNC', 'CTE']:

        if mode == 'UNC':
            lfps = likely_fps_uncertainty
        elif mode == 'CTE':
            lfps = likely_fps_cte
        else:
            lfps = likely_fps_common

        ''' 
            2. compute improvement set
        '''
        x_train, x_test = load_data_for_vae_retraining(cfg, sampling=15)
        improvement_set = load_improvement_set(cfg, lfps)

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

        # for improvement_ratio in [2, 5, 10]:
        for improvement_ratio in [2]:
            print("Using improvement ratio: " + str(improvement_ratio))
            for i in range(improvement_ratio - 1):
                temp = initial_improvement_set[:]
                improvement_set = np.concatenate((temp, improvement_set), axis=0)

            x_train_improvement_set, x_test_improvement_set = train_test_split(improvement_set, test_size=cfg.TEST_SIZE,
                                                                               random_state=0)

            x_train = np.concatenate((x_train, x_train_improvement_set), axis=0)
            x_test = np.concatenate((x_test, x_test_improvement_set), axis=0)

            print("New training data set: " + str(len(x_train)) + " elements")

            ''' 
                3. retrain using GAUSS's configuration
            '''
            weights = None

            newname = name + '-CI-RETRAINED-' + str(improvement_ratio) + "X-" + mode
            train_vae_model(cfg, vae, newname, x_train, x_test, delete_model=True, retraining=True,
                            sample_weights=weights)

            encoder = tensorflow.keras.models.load_model(cfg.SAO_MODELS_DIR + '/' + 'encoder-' + newname)
            decoder = tensorflow.keras.models.load_model(cfg.SAO_MODELS_DIR + '/' + 'decoder-' + newname)
            print("loaded retrained VAE from disk")

            vae = VAE(model_name=newname,
                      loss=cfg.LOSS_SAO_MODEL,
                      latent_dim=cfg.SAO_LATENT_DIM,
                      encoder=encoder,
                      decoder=decoder)
            vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE))

            ''' 
                4. evaluate retrained model (GAUSS)  
            '''
            new_losses = load_or_compute_losses(vae, dataset, newname, delete_cache=True)
            plot_reconstruction_losses(original_losses, new_losses, newname, threshold_nominal, None)
            get_scores(cfg, newname, original_losses, new_losses, threshold_nominal)

        ''' 
            5. load data for retraining
        '''
        x_train, x_test = load_data_for_vae_retraining(cfg, sampling=1)
        improvement_set = load_improvement_set(cfg, lfps)

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

        temp = initial_improvement_set[:]
        improvement_set = np.concatenate((temp, improvement_set), axis=0)

        x_train_improvement_set, x_test_improvement_set = train_test_split(improvement_set, test_size=cfg.TEST_SIZE,
                                                                           random_state=0)

        x_train = np.concatenate((x_train, x_train_improvement_set), axis=0)
        x_test = np.concatenate((x_test, x_test_improvement_set), axis=0)

        ''' 
            6. retrain using JSEP's configuration
        '''
        # magic happens here
        weights = np.array(original_losses)

        vae, name = load_vae(cfg, load_vae_from_disk=True)
        newname = name + '-CI-RETRAINED-JSEP-' + mode
        train_vae_model(cfg, vae, newname, x_train, x_test, delete_model=True, retraining=True, sample_weights=weights)

        encoder = tensorflow.keras.models.load_model(cfg.SAO_MODELS_DIR + '/' + 'encoder-' + newname)
        decoder = tensorflow.keras.models.load_model(cfg.SAO_MODELS_DIR + '/' + 'decoder-' + newname)
        print("loaded retrained VAE from disk")

        vae = VAE(model_name=newname,
                  loss=cfg.LOSS_SAO_MODEL,
                  latent_dim=cfg.SAO_LATENT_DIM,
                  encoder=encoder,
                  decoder=decoder)
        vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.SAO_LEARNING_RATE))

        ''' 
            7. evaluate retrained (JSEP) 
        '''
        new_losses = load_or_compute_losses(vae, dataset, newname, delete_cache=True)
        plot_reconstruction_losses(original_losses, new_losses, newname, threshold_nominal, None)
        get_scores(cfg, newname, original_losses, new_losses, threshold_nominal)


def main():
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    evaluate_class_imbalance(cfg)


if __name__ == '__main__':
    main()
