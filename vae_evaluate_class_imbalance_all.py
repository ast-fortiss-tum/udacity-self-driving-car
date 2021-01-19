from config import Config
from vae_evaluate_class_imbalance import evaluate_class_imbalance

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    only_center_images = [True, False]
    use_crop = [True, False]
    loss_func = ["VAE", "MSE"]
    latent_space = [2, 16]

    for only in only_center_images:
        cfg.USE_ONLY_CENTER_IMG = only
        for crop in use_crop:
            cfg.USE_CROP = crop
            for loss in loss_func:
                cfg.LOSS_SAO_MODEL = loss
                for ld in latent_space:
                    cfg.SAO_LATENT_DIM = ld
                    evaluate_class_imbalance(cfg)
