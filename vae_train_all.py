import utils
from config import Config
from utils_vae import load_data_for_vae_training, load_vae
from vae_train import train_vae_model
from vae_evaluate_class_imbalance import evaluate_class_imbalance

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    latent_space = [16]
    loss_func = ["MSE", "VAE"]
    use_crop = [False, True]  # True gives poor results

    for ld in latent_space:
        cfg.SAO_LATENT_DIM = ld
        for loss in loss_func:
            cfg.LOSS_SAO_MODEL = loss
            for crop in use_crop:
                cfg.USE_CROP = crop

                drive = utils.get_driving_styles(cfg)
                x_train, x_test = load_data_for_vae_training(cfg)

                vae, name = load_vae(cfg, load_vae_from_disk=False)
                train_vae_model(cfg, vae, name, x_train, x_test, delete_model=False, retraining=False,
                                sample_weights=None)

    for ld in latent_space:
        cfg.SAO_LATENT_DIM = ld
        for loss in loss_func:
            cfg.LOSS_SAO_MODEL = loss
            for crop in use_crop:
                cfg.USE_CROP = crop

                # Gauss's setting
                cfg.IMPROVEMENT_RATIO = 20
                evaluate_class_imbalance(cfg)

                # JSEP's setting
                cfg.IMPROVEMENT_RATIO = 1
                evaluate_class_imbalance(cfg)
