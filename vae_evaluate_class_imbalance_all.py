from config import Config
from vae_evaluate_class_imbalance import evaluate_class_imbalance

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    # latent_space = [2, 16]
    # loss_func = ["MSE", "VAE"]
    # only_center_images = [True, False]
    # use_crop = [False, True]

    latent_space = [8]
    loss_func = ["MSE", "VAE"]
    use_crop = [False, True]

    for ld in latent_space:
        cfg.SAO_LATENT_DIM = ld
        for loss in loss_func:
            cfg.LOSS_SAO_MODEL = loss
            for crop in use_crop:
                cfg.USE_CROP = crop
                evaluate_class_imbalance(cfg)
