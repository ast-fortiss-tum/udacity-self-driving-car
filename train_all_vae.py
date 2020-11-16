import utils
from config import Config
from train_vae import load_data_for_vae, run_training

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("myconfig.py")

    tracks = cfg.TRACK
    drive = utils.get_driving_styles(cfg)

    only_center_images = [True, False]
    use_crop = [True, False]
    loss_func = ["VAE", "MSE"]

    x_train, x_test = load_data_for_vae(cfg)

    for track in tracks:
        for only in only_center_images:
            for crop in use_crop:
                for loss in loss_func:
                    run_training(cfg, x_test, x_train)
