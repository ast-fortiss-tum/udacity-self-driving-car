# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import glob
import os

from config import Config
from heatmap import compute_heatmap

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    simulations = glob.glob('simulations/*')

    for sim in simulations:
        compute_heatmap(cfg, simulation_name=sim)
