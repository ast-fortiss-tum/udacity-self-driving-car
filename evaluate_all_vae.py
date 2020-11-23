import utils
from config import Config
from evaluate_vae import load_all_images, load_and_eval_vae

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("myconfig.py")

    tracks = cfg.TRACK
    drive = utils.get_driving_styles(cfg)

    only_center_images = [True, False]
    use_crop = [True, False]
    loss_func = ["MSE"]  # ["VAE", "MSE"]

    data = load_all_images(cfg)

    for track in tracks:
        cfg.TRACK = track
        for only in only_center_images:
            cfg.USE_ONLY_CENTER_IMG = only
            for crop in use_crop:
                cfg.USE_CROP = crop
                for loss in loss_func:
                    cfg.LOSS_SAO_MODEL = loss
                    load_and_eval_vae(cfg, data)
