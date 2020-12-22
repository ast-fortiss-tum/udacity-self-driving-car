import numpy as np

from config import Config
from utils import load_improvement_set
from utils_vae import load_data_for_vae_training, load_vae
from vae_train import train_vae_model


def main():
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    lfp_unc = np.load('lfp_unc.npy')
    x_train, x_test = load_data_for_vae_training(cfg)
    improvement_set = load_improvement_set(cfg, lfp_unc)

    x_train = improvement_set

    for i in range(10):
        x_train_new = np.concatenate((x_train, improvement_set), axis=0)
        x_train = x_train_new
    print("Improvement data set: " + str(len(x_train)) + " elements")

    vae, name = load_vae(cfg, load_vae_from_disk=True)
    train_vae_model(cfg, vae, name, x_train, x_test, delete_model=False, retraining=True)


if __name__ == '__main__':
    main()
