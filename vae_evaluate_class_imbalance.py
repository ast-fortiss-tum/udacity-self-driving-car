import numpy as np

from config import Config
from utils import load_all_images
from utils import plot_reconstruction_losses, load_improvement_set
from utils_vae import load_vae, load_data_for_vae_retraining
from vae_evaluate import load_or_compute_losses, get_threshold, get_scores
from vae_train import train_vae_model


def main():
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    # 1. compute reconstruction error on nominal images
    dataset = load_all_images(cfg)
    vae, name = load_vae(cfg, load_vae_from_disk=True)
    losses = load_or_compute_losses(vae, dataset, name, delete_cache=False)
    threshold_nominal = get_threshold(losses, conf_level=0.95)
    plot_reconstruction_losses(losses, name, threshold_nominal)
    lfp_unc, lfp_cte = get_scores(cfg, losses)
    np.save('lfp_unc_before.npy', lfp_unc)
    np.save('lfp_cte_before.npy', lfp_cte)

    # 2. compute improvement set and retrain
    x_train, x_test = load_data_for_vae_retraining(cfg)
    improvement_set = load_improvement_set(cfg, lfp_unc)

    print("Old training data set: " + str(len(x_train)) + " elements")
    print("Improvement data set: " + str(len(improvement_set)) + " elements")

    for i in range(cfg.IMPROVEMENT_RATIO):
        x_train_new = np.concatenate((x_train, improvement_set), axis=0)
        x_train = x_train_new

    print("New training data set: " + str(len(x_train)) + " elements")

    cfg.BATCH_SIZE = 16

    name = name + '-RETRAINED-' + str(cfg.IMPROVEMENT_RATIO) + "X"
    train_vae_model(cfg, vae, name, x_train, x_test, delete_model=False, retraining=True)

    # 3. evaluate w/ old threshold
    new_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)
    # threshold_nominal = get_threshold(new_losses, conf_level=0.95)
    plot_reconstruction_losses(new_losses, name, threshold_nominal)
    get_scores(cfg, new_losses)


if __name__ == '__main__':
    main()
