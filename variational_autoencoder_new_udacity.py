import os

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tqdm import tqdm

import utils
from config import Config
from evaluate_vae import load_all_images, plot_pictures_orig_rec
from train_vae import load_data_for_vae
from utils import RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS
from vae_batch_generator import Generator
from variational_autoencoder import normalize_and_reshape

original_dim = RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS


class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):

    def call(self, inputs):
        latent_dim = 2

        encoder_inputs = keras.Input(shape=(original_dim,))
        x = Dense(512, activation='relu')(encoder_inputs)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs=encoder_inputs, outputs=[z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # TODO: give it a try to a VAE with convolutional layers
        # x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        # x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Flatten()(x)
        # x = layers.Dense(16, activation="relu")(x)

        return encoder


class Decoder(layers.Layer):

    def call(self, latent_inputs):
        latent_inputs = keras.Input(shape=(2,), name='z_sampling')
        x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001))(latent_inputs)
        decoder_outputs = Dense(original_dim, activation='sigmoid')(x)

        # TODO: give it a try to a VAE with convolutional layers
        # x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        # x = layers.Reshape((7, 7, 64))(x)
        # x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        # x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        # decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

        decoder = keras.Model(inputs=latent_inputs, outputs=decoder_outputs, name="decoder")
        decoder.summary()

        return decoder


# Define the VAE as a `Model` with a custom `train_step`
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data, reconstruction)
            )
            reconstruction_loss *= RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


"""
## Train the VAE
"""

cfg = Config()
cfg.from_pyfile("myconfig.py")

x_train, x_test = load_data_for_vae(cfg)
udacity_images = np.concatenate([x_train, x_test], axis=0)

if not os.path.exists('udacity-vae-encoder'):
    encoder = Encoder().call(RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS)
    decoder = Decoder().call((2,))

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

    es = keras.callbacks.EarlyStopping(monitor='loss', patience=5, mode="auto", restore_best_weights=True)

    x_train = shuffle(x_train, random_state=0)
    x_test = shuffle(x_test, random_state=0)
    train_generator = Generator(vae, x_train, True, cfg)
    val_generator = Generator(vae, x_test, True, cfg)

    history = vae.fit(train_generator,
                      validation_data=val_generator,
                      shuffle=False,
                      epochs=5,
                      callbacks=[es],
                      # steps_per_epoch=len(x_train) // cfg.BATCH_SIZE,
                      verbose=1)

    encoder.save('udacity-vae-encoder')
    decoder.save('udacity-vae-decoder')

else:

    encoder = keras.models.load_model('udacity-vae-encoder')
    decoder = keras.models.load_model('udacity-vae-decoder')

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

udacity_images = load_all_images(cfg)

i = 0
for x in tqdm(udacity_images):
    i = i + 1

    if i % 100 == 0:
        x = utils.resize(x)
        x = normalize_and_reshape(x)

        z_mean, z_log_var, z = encoder.predict(x)
        decoded = decoder.predict(z)

        loss = vae.test_on_batch(x)

        plot_pictures_orig_rec(x, decoded, None, loss)

# def plot_latent(decoder):
#     # display a n*n 2D manifold of digits
#     n = 30
#     digit_size = 28
#     scale = 2.0
#     figsize = 15
#     figure = np.zeros((digit_size * n, digit_size * n))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-scale, scale, n)
#     grid_y = np.linspace(-scale, scale, n)[::-1]
#
#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.array([[xi, yi]])
#             x_decoded = decoder.predict(z_sample)
#             digit = x_decoded[0].reshape(digit_size, digit_size)
#             figure[
#             i * digit_size: (i + 1) * digit_size,
#             j * digit_size: (j + 1) * digit_size,
#             ] = digit
#
#     plt.figure(figsize=(figsize, figsize))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure, cmap="Greys_r")
#     plt.show()
#
#
# plot_latent(decoder)
#
# """
# ## Display how the latent space clusters different digit classes
# """
#
#
# def plot_label_clusters(encoder, data, labels):
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _ = encoder.predict(data)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
#     plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.show()
#
#
# (x_train, y_train), _ = keras.datasets.mnist.load_data()
# x_train = np.expand_dims(x_train, -1).astype("float32") / 255
#
# plot_label_clusters(encoder, x_train, y_train)
