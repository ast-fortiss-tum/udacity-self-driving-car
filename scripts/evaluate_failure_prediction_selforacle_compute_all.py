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
from evaluate_failure_prediction_selforacle import evaluate_failure_prediction

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    for num_samples in ['all', 1000, 1100, 1200, 1300, 1400, 1500]:
        for condition in ['ood', 'mutants']:
            simulations = natsorted(glob.glob('simulations/' + condition + '/*'))
            for am in ['mean', 'max']:
                for sim in simulations:
                    sim = sim.replace("simulations/", "")
                    if "nominal" not in sim:
                        evaluate_failure_prediction(cfg,
                                                    simulation_name=sim,
                                                    aggregation_method=am,
                                                    condition=condition,
                                                    num_samples=num_samples)
