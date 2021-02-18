import os

import pandas as pd

import utils_vae
from config import Config
from utils import load_all_images
from vae_evaluate import load_or_compute_losses, get_results_mispredictions
from keras import backend as K
import gc


def evaluate_novelty_detection(cfg, track, condition, metric, technique):
    """
        1. compute reconstruction error on nominal images
        and compute the likely false positives
    """

    # 1. recompute the nominal threshold
    cfg.SIMULATION_NAME = 'gauss-journal-' + track + '-nominal'
    dataset = load_all_images(cfg)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df_nominal = pd.read_csv(path)

    if cfg.USE_ONLY_CENTER_IMG:
        name = cfg.TRACK + '-' + cfg.LOSS_SAO_MODEL + 'loss' + "-latent" + str(cfg.SAO_LATENT_DIM) \
               + '-centerimg-' + 'nocrop' + technique + metric
    else:
        name = cfg.TRACK + '-' + cfg.LOSS_SAO_MODEL + 'loss' + "-latent" + str(cfg.SAO_LATENT_DIM) \
               + '-allimg-' + 'nocrop' + technique + metric

    vae = utils_vae.load_vae_by_name(name)

    original_losses = load_or_compute_losses(vae, dataset, name, delete_cache=False)

    # 2. evaluate on novel conditions (rain)
    cfg.SIMULATION_NAME = 'gauss-journal-' + track + condition
    dataset = load_all_images(cfg)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df_anomalous = pd.read_csv(path)

    name = cfg.SIMULATION_NAME
    new_losses = load_or_compute_losses(vae, dataset, name, delete_cache=False)

    for seconds in range(1, 4):  # 1, 2, 3
        get_results_mispredictions(cfg, name,
                                   original_losses, new_losses,
                                   data_df_nominal, data_df_anomalous,
                                   seconds)

    del vae
    K.clear_session()
    gc.collect()


def main():
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    # condition = '-rain'
    # metric = "-UNC"
    # technique = "-CI-RETRAINED-2X"
    #
    # evaluate_novelty_detection(cfg, cfg.TRACK, condition, metric, technique)


if __name__ == '__main__':
    main()
