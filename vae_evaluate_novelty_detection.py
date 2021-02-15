import os

import pandas as pd

import utils_vae
from config import Config
from utils import load_all_images
from utils import plot_reconstruction_losses
from vae_evaluate import load_or_compute_losses, get_threshold, get_scores


def evaluate_novelty_detection(cfg):
    """
        1. compute reconstruction error on nominal images
        and compute the likely false positives
    """

    # 1. recompute the nominal threshold
    cfg.SIMULATION_NAME = 'gauss-journal-track1-nominal'
    dataset = load_all_images(cfg)

    name = 'track1-VAEloss-latent2-centerimg-nocrop-CI-RETRAINED-JSEP-CTE'
    vae = utils_vae.load_vae_by_name(name)

    original_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)
    threshold_nominal = get_threshold(original_losses, conf_level=0.95)

    # 2. evaluate on novel conditions
    cfg.SIMULATION_NAME = 'gauss-journal-track1-dave2-novelty-detection-daynight'
    dataset = load_all_images(cfg)

    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)

    name = cfg.SIMULATION_NAME
    new_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)
    plot_reconstruction_losses(new_losses, None, name, threshold_nominal, None, data_df)
    likely_fps_uncertainty, likely_fps_cte, _ = get_scores(cfg, name, new_losses, new_losses,
                                                           threshold_nominal)


def main():
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    evaluate_novelty_detection(cfg)


if __name__ == '__main__':
    main()
