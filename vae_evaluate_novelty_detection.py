import os

import tensorflow
from tensorflow import keras

import utils_vae
from config import Config
from utils import load_all_images
from utils import plot_reconstruction_losses
from vae import VAE
from vae_evaluate import load_or_compute_losses, get_threshold, get_scores


def evaluate_novelty_detection(cfg):
    ''' 
        1. compute reconstruction error on nominal images
        and compute the likely false positives
    '''
    cfg.SIMULATION_NAME = 'gauss-journal-track1-dave2-nominal-latent2'
    dataset = load_all_images(cfg)

    name = 'track1-VAEloss-latent2-centerimg-nocrop-CI-RETRAINED-2X-UNC'
    vae = utils_vae.load_vae_by_name(name)

    original_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)
    threshold_nominal = get_threshold(original_losses, conf_level=0.95)
    likely_fps_uncertainty, likely_fps_cte, _ = get_scores(cfg, name, original_losses, original_losses,
                                                           threshold_nominal)

    cfg.SIMULATION_NAME = 'gauss-journal-track1-dave2-road-asphalt-dirt'
    dataset = load_all_images(cfg)
    new_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)
    plot_reconstruction_losses(new_losses, new_losses, name, threshold_nominal, None)
    likely_fps_uncertainty, likely_fps_cte, _ = get_scores(cfg, name, new_losses, new_losses,
                                                           threshold_nominal)


def main():
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    evaluate_novelty_detection(cfg)


if __name__ == '__main__':
    main()
