import numpy as np

from variational_autoencoder import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, normalize_and_reshape
from utils import load_image
from tensorflow.python.keras.utils.data_utils import Sequence


class Generator(Sequence):

    def __init__(self, model, path_to_pictures, is_training, cfg):
        self.model = model
        self.path_to_pictures = path_to_pictures
        self.is_training = is_training
        self.cfg = cfg

    def __getitem__(self, index):
        start_index = index * self.cfg.BATCH_SIZE
        end_index = start_index + self.cfg.BATCH_SIZE
        batch_paths = self.path_to_pictures[start_index:end_index]

        images = np.empty([len(batch_paths), IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS])
        for i, paths in enumerate(batch_paths):

            if self.cfg.USE_ONLY_CENTER_IMG:
                center = batch_paths[i]
            else:
                center, left, right = batch_paths[i]

            # TODO: add the left and right image as well
            image = load_image(self.cfg.SIMULATION_DATA_DIR, center)
            image = normalize_and_reshape(image)
            images[i] = image
        return images, images

    def __len__(self):
        return len(self.path_to_pictures) // self.cfg.BATCH_SIZE
