import utils
from config import Config
from utils_vae import load_data_for_vae_training, load_vae
from vae_train import train_vae_model

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    only_center_images = [True]
    use_crop = [False]
    loss_func = ["MSE"]
    latent_space = [2]

    for only in only_center_images:
        cfg.USE_ONLY_CENTER_IMG = only
        assert cfg.USE_ONLY_CENTER_IMG == only
        for crop in use_crop:
            cfg.USE_CROP = crop
            assert cfg.USE_CROP == crop
            for loss in loss_func:
                cfg.LOSS_SAO_MODEL = loss
                assert cfg.LOSS_SAO_MODEL == loss
                for ld in latent_space:
                    cfg.SAO_LATENT_DIM = ld
                    assert cfg.SAO_LATENT_DIM == ld

                    drive = utils.get_driving_styles(cfg)
                    x_train, x_test = load_data_for_vae_training(cfg)

                    vae, name = load_vae(cfg, load_vae_from_disk=False)
                    train_vae_model(cfg, vae, name, x_train, x_test, delete_model=False, retraining=False)
