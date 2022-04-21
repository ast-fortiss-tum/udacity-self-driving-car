# Copyright 2021 by Andrea Stocco, the Software Institute at USI.
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME.
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import glob
import os

from natsort import natsorted

from config import Config
from evaluate_failure_prediction_vae_on_heatmaps import evaluate_failure_prediction

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    simulations = glob.glob('simulations/*')

    # cfg.TESTING_DATA_DIR = "/Volumes/Seagate Backup Plus Drive/Heatmaps-Failure-Prediction"
    simulations = natsorted(glob.glob(cfg.TESTING_DATA_DIR + '/*'))

    for ht in ['smoothgrad']:
        for am in ['mean', 'max']:
            for sim in simulations:
                if "fog" in sim:
                    sim = sim.replace("simulations/", "")
                    evaluate_failure_prediction(cfg,
                                                heatmap_type=ht,
                                                simulation_name=sim,
                                                aggregation_method=am)
                else:
                    continue
