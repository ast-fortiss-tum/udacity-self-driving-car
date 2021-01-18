"""
@author: astocco
"""

import random

import matplotlib.image as mpimg
import numpy as np
from tensorflow.keras.utils import Sequence

from utils import RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS, resize, crop
from vae import normalize_and_reshape


class Generator(Sequence):

    def __init__(self, path_to_pictures, is_training, cfg, sample_weight):
        self.path_to_pictures = path_to_pictures
        self.is_training = is_training
        self.cfg = cfg
        self.sample_weight = sample_weight

    def __getitem__(self, index):
        start_index = index * self.cfg.SAO_BATCH_SIZE
        end_index = start_index + self.cfg.SAO_BATCH_SIZE
        batch_paths = self.path_to_pictures[start_index:end_index]
        weights = self.sample_weight[start_index:end_index]

        images = np.empty([len(batch_paths), RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS])
        for i, paths in enumerate(batch_paths):

            if self.cfg.USE_ONLY_CENTER_IMG:
                center = batch_paths[i][0]  # select the center image from the batch
                # image = load_image(self.cfg.TRAINING_SET_DIR, center)
                try:
                    image = mpimg.imread(self.cfg.TRAINING_DATA_DIR + "/" + self.cfg.TRAINING_SET_DIR + center)
                except FileNotFoundError:
                    image = mpimg.imread(center)
            else:
                center, left, right = batch_paths[i]

                choices = [0, 1, 2]  # 0=center, 1=left, 2=right
                choice = random.choice(choices)
                if choice == 0:
                    try:
                        image = mpimg.imread(self.cfg.TRAINING_DATA_DIR + "/" + self.cfg.TRAINING_SET_DIR + center)
                    except FileNotFoundError:
                        image = mpimg.imread(center)
                elif choice == 1:
                    try:
                        image = mpimg.imread(self.cfg.TRAINING_DATA_DIR + "/" + self.cfg.TRAINING_SET_DIR + left)
                    except FileNotFoundError:
                        image = mpimg.imread(left)
                elif choice == 2:
                    try:
                        image = mpimg.imread(self.cfg.TRAINING_DATA_DIR + "/" + self.cfg.TRAINING_SET_DIR + right)
                    except FileNotFoundError:
                        image = mpimg.imread(right)
                else:
                    print('wrong image index in vae_batch_generator. Using default\'s 0 (center)')
                    try:
                        image = mpimg.imread(self.cfg.TRAINING_DATA_DIR + "/" + self.cfg.TRAINING_SET_DIR + center)
                    except FileNotFoundError:
                        image = mpimg.imread(center)

            if self.cfg.USE_CROP:
                image = crop(image)

            image = resize(image)

            # visualize whether the input image as expected
            # import matplotlib.pyplot as plt
            # plt.imshow(image)
            # plt.show()

            image = normalize_and_reshape(image)
            images[i] = image

        return images, images, weights

    def __len__(self):
        return len(self.path_to_pictures) // self.cfg.SAO_BATCH_SIZE
