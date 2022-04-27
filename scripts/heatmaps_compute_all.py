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

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    simulations = glob.glob('simulations/mutants/*')

    cfg.SDC_MODEL_NAME = "track1-dave2-uncropped-mc-034.h5"

    for sim in simulations:
        sim = sim.replace("simulations/", "")
        # if "rain" in sim or "snow" in sim or "fog" in sim:
        if "nominal" not in sim:
            compute_heatmap(cfg, simulation_name=sim, attention_type="GradCam++")
