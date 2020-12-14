import utils
from config import Config
from vae_evaluate import load_all_images, load_and_eval_vae

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    drive = utils.get_driving_styles(cfg)

    only_center_images = [True, False]
    use_crop = [True, False]
    loss_func = ["VAE", "MSE"]
    latent_space = [2, 4, 8, 16, 32, 64, 128, 256]
    intermediate_space = [512]

    data = load_all_images(cfg)

    for only in only_center_images:
        cfg.USE_ONLY_CENTER_IMG = only
        for crop in use_crop:
            cfg.USE_CROP = crop
            for loss in loss_func:
                cfg.LOSS_SAO_MODEL = loss
                for ld in latent_space:
                    cfg.SAO_LATENT_DIM = ld
                    for intdim in intermediate_space:
                        cfg.SAO_INTERMEDIATE_DIM = intdim
                        load_and_eval_vae(cfg, data, delete_cache=True)
