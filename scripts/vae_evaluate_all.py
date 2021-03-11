# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import os

from config import Config
from vae_evaluate import load_all_images, load_and_eval_vae

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    tracks = ["track1", "track2", "track3"]

    loss_func = ["MSE", "VAE"]
    latent_space = [2, 4, 8, 16]

    data = load_all_images(cfg)

    for t in tracks:
        cfg.TRACK = t
        for loss in loss_func:
            cfg.LOSS_SAO_MODEL = loss
            for ld in latent_space:
                cfg.SAO_LATENT_DIM = ld
                load_and_eval_vae(cfg, data, delete_cache=True)
