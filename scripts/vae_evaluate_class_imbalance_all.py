# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
from config import Config
from vae_evaluate_class_imbalance import evaluate_class_imbalance

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    tracks = ["track1", "track2", "track3"]

    latent_space = [2, 4, 8, 16]
    loss_func = ["VAE"]
    cfg.NUM_EPOCHS_SAO_MODEL = 50

    for t in tracks:
        cfg.TRACK = t
        for ld in latent_space:
            cfg.SAO_LATENT_DIM = ld
            for loss in loss_func:
                cfg.LOSS_SAO_MODEL = loss
                evaluate_class_imbalance(cfg)
