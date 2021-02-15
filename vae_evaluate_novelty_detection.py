import os

import pandas as pd

import utils_vae
from config import Config
from utils import load_all_images
from vae_evaluate import load_or_compute_losses, get_threshold, get_scores_mispredictions


def evaluate_novelty_detection(cfg, track, condition, metric, technique):
    """
        1. compute reconstruction error on nominal images
        and compute the likely false positives
    """

    # 1. recompute the nominal threshold
    cfg.SIMULATION_NAME = 'gauss-journal-' + track + '-nominal'
    dataset = load_all_images(cfg)

    name = cfg.TRACK + '-' + cfg.LOSS_SAO_MODEL + 'loss' + "-latent" + str(cfg.SAO_LATENT_DIM) \
           + cfg.USE_ONLY_CENTER_IMG + cfg.USE_CROP + technique + metric

    vae = utils_vae.load_vae_by_name(name)

    original_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)
    threshold_nominal = get_threshold(original_losses, conf_level=0.95)

    # 2. evaluate on novel conditions (rain)
    cfg.SIMULATION_NAME = 'gauss-journal-' + track + condition
    dataset = load_all_images(cfg)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)

    name = cfg.SIMULATION_NAME
    new_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)

    for seconds in range(1, 6):  # 1, 2, 3, 4, 5
        get_scores_mispredictions(cfg, name, new_losses, data_df, threshold_nominal, seconds)


def main():
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    condition = 'alsphalt'
    metric = "-UNC"
    technique = "-CI-RETRAINED-2X"

    evaluate_novelty_detection(cfg, cfg.TRACK, condition, metric, technique)


if __name__ == '__main__':
    main()
