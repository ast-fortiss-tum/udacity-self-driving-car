import numpy as np

from config import Config
from self_driving_car_train import load_data, train_model
from utils_models import *

np.random.seed(0)


def main():
    """
    Load train/validation data_nominal set and train the model
    """
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    x_train, x_test, y_train, y_test = load_data(cfg)

    models = ["dave2"]
    use_mc = [True, False]

    for m in models:
        cfg.SDC_MODEL_NAME = m
        for unc in use_mc:
            cfg.USE_PREDICTIVE_UNCERTAINTY = unc
            model = build_model(cfg.SDC_MODEL_NAME, cfg.USE_PREDICTIVE_UNCERTAINTY)
            train_model(model, cfg, x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
