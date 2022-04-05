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

from selforacle import utils_vae
from config import Config
from utils import load_all_images
from vae_evaluate import load_or_compute_losses, get_results_mispredictions


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
        name = cfg.TRACK + '-' + cfg.LOSS_SAO_MODEL + "-latent" + str(cfg.SAO_LATENT_DIM) + technique + metric
    else:
        name = cfg.TRACK + '-' + cfg.LOSS_SAO_MODEL + "-latent" + str(cfg.SAO_LATENT_DIM) + technique + metric

    vae = utils_vae.load_vae_by_name(name)

    original_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)

    # 2. evaluate on novel conditions (rain)
    cfg.SIMULATION_NAME = 'gauss-journal-' + track + condition
    dataset = load_all_images(cfg)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df_anomalous = pd.read_csv(path)

    new_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)

    sim_name = cfg.SIMULATION_NAME

    for seconds in range(1, 4):  # 1, 2, 3
        get_results_mispredictions(cfg, sim_name, name,
                                   original_losses, new_losses,
                                   data_df_nominal, data_df_anomalous,
                                   seconds)

    del vae
    K.clear_session()
    gc.collect()


def main():
    os.chdir(os.getcwd().replace('scripts', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    # condition = '-rain'
    # metric = "-UNC"
    # technique = "-CI-RETRAINED-2X"
    #
    # evaluate_novelty_detection(cfg, cfg.TRACK, condition, metric, technique)


if __name__ == '__main__':
    main()
