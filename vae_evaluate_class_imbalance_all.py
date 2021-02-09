from config import Config
from vae_evaluate_class_imbalance import evaluate_class_imbalance

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    latent_space = [16]
    loss_func = ["VAE"]
    use_only_center_image = [False]

    use_crop = [False]
    cfg.NUM_EPOCHS_SAO_MODEL = 50

    for ld in latent_space:
        cfg.SAO_LATENT_DIM = ld
        for loss in loss_func:
            cfg.LOSS_SAO_MODEL = loss
            for input_image in use_only_center_image:
                cfg.USE_ONLY_CENTER_IMG = input_image
                for crop in use_crop:
                    cfg.USE_CROP = crop
                    evaluate_class_imbalance(cfg)
