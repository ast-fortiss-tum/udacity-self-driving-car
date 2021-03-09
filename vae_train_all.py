from config import Config
from utils_vae import load_data_for_vae_training, load_vae
from vae_train import train_vae_model

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    tracks = ["track1"]  #, "track2", "track3"]

    latent_space = [2, 4, 8, 16]
    loss_func = ["MSE", "VAE"]
    use_only_center_image = [True, False]

    use_crop = [False]
    cfg.NUM_EPOCHS_SAO_MODEL = 100

    for t in tracks:
        cfg.TRACK = t
        for ld in latent_space:
            cfg.SAO_LATENT_DIM = ld
            for loss in loss_func:
                cfg.LOSS_SAO_MODEL = loss
                for input_image in use_only_center_image:
                    cfg.USE_ONLY_CENTER_IMG = input_image
                    for crop in use_crop:
                        cfg.USE_CROP = crop

                        x_train, x_test = load_data_for_vae_training(cfg)

                        vae, name = load_vae(cfg, load_vae_from_disk=False)
                        train_vae_model(cfg, vae, name, x_train, x_test, delete_model=True, retraining=False,
                                        sample_weights=None)
