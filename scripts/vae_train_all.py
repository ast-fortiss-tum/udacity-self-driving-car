# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
from config import Config
from selforacle.utils_vae import load_data_for_vae_training, load_vae
from selforacle.vae_train import train_vae_model
import os

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    tracks = ["track1", "track2", "track3"]

    latent_space = [2, 4, 8, 16]
    loss_func = ["MSE", "VAE"]
    cfg.NUM_EPOCHS_SAO_MODEL = 5

    for t in tracks:
        cfg.TRACK = t
        for ld in latent_space:
            cfg.SAO_LATENT_DIM = ld
            for loss in loss_func:
                cfg.LOSS_SAO_MODEL = loss

                x_train, x_test = load_data_for_vae_training(cfg)

                vae, name = load_vae(cfg, load_vae_from_disk=False)
                train_vae_model(cfg, vae, name, x_train, x_test,
                                delete_model=True,
                                retraining=False,
                                sample_weights=None)
