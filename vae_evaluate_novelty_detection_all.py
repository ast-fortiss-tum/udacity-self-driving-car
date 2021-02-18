from config import Config
from vae_evaluate_novelty_detection import evaluate_novelty_detection

if __name__ == '__main__':
    cfg = Config()
    cfg.from_pyfile("config_my.py")

    tracks = ["-track1", "-track2", "-track3"]
    conditions = ["-rain"]

    latent_space = [2, 4, 8, 16]
    loss_func = ["MSE", "VAE"]
    use_only_center_image = [True, False]
    infieldmetrics = ["-UNC", "-CTE"]
    techniques = ["-CI-RETRAINED-2X", "-CI-RETRAINED-JSEP"]

    for t in tracks:
        cfg.TRACK = t
        for c in conditions:
            for ifm in infieldmetrics:
                for tech in techniques:
                    for ld in latent_space:
                        cfg.SAO_LATENT_DIM = ld
                        for loss in loss_func:
                            cfg.LOSS_SAO_MODEL = loss
                            for input_image in use_only_center_image:
                                cfg.USE_ONLY_CENTER_IMG = input_image
                                cfg.USE_CROP = False
                                evaluate_novelty_detection(cfg, t, c, ifm, tech)
