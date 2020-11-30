import os

import matplotlib.pyplot as plt
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
USE_MSE = False


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
    def __init__(self, encoder, decoder, use_mse, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.use_mse = use_mse

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(keras.losses.mean_squared_error(data, reconstruction))
            reconstruction_loss *= RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT

            if not self.use_mse:
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
            else:
                total_loss = reconstruction_loss
                grads = tape.gradient(total_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                return {
                    "loss": total_loss,
                    "reconstruction_loss": reconstruction_loss,
                }

    def call(self, inputs):
        z_mean, z_log_var, z = encoder(inputs)
        reconstruction = decoder(z)
        reconstruction_loss = tf.reduce_mean(keras.losses.mean_squared_error(inputs, reconstruction))
        reconstruction_loss *= RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT

        if not self.use_mse:
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
            self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
            self.add_metric(total_loss, name='total_loss', aggregation='mean')
            self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
            return reconstruction
        else:
            total_loss = reconstruction_loss
            self.add_metric(total_loss, name='total_loss', aggregation='mean')
            self.add_metric(reconstruction_loss, name='reconstruction_loss', aggregation='mean')
            return reconstruction


"""
## Train the VAE
"""

cfg = Config()
cfg.from_pyfile("myconfig.py")

x_train, x_test = load_data_for_vae(cfg)
udacity_images = np.concatenate([x_train, x_test], axis=0)

if not os.path.exists('udacity-vae-encoder') or not os.path.exists('udacity-mse-encoder'):
    encoder = Encoder().call(RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS)
    decoder = Decoder().call((2,))

    vae = VAE(encoder, decoder, use_mse=USE_MSE)
    vae.compile(optimizer=keras.optimizers.Adam())

    # es = keras.callbacks.EarlyStopping(monitor='loss', patience=5, mode="auto", restore_best_weights=True)

    x_train = shuffle(x_train, random_state=0)
    x_test = shuffle(x_test, random_state=0)
    train_generator = Generator(vae, x_train, True, cfg)
    val_generator = Generator(vae, x_test, True, cfg)

    history = vae.fit(train_generator,
                      validation_data=val_generator,
                      shuffle=False,
                      epochs=5,
                      # callbacks=[es],
                      # steps_per_epoch=len(x_train) // cfg.BATCH_SIZE,
                      verbose=1)

    if not vae.use_mse:
        encoder.save('udacity-vae-encoder')
        decoder.save('udacity-vae-decoder')

        plt.plot(history.history['reconstruction_loss'])
        plt.plot(history.history['kl_loss'])
        plt.plot(history.history['val_reconstruction_loss'])
        plt.plot(history.history['val_kl_loss'])
        plt.ylabel('reconstruction loss new udacity MSE loss')
        plt.xlabel('epoch')
        plt.title('training')
        plt.legend(['reconstruction_loss', 'kl_loss', 'val_reconstruction_loss', 'val_kl_loss'], loc='upper left')
        plt.show()
    else:
        encoder.save('udacity-mse-encoder')
        decoder.save('udacity-mse-decoder')

        plt.plot(history.history['reconstruction_loss'])
        plt.plot(history.history['val_reconstruction_loss'])
        plt.ylabel('reconstruction loss new udacity VAE loss')
        plt.xlabel('epoch')
        plt.title('training')
        plt.legend(['reconstruction_loss', 'val_reconstruction_loss'], loc='upper left')
        plt.show()

else:
    if USE_MSE:
        encoder = keras.models.load_model('udacity-mse-encoder')
        decoder = keras.models.load_model('udacity-mse-decoder')
    else:
        encoder = keras.models.load_model('udacity-vae-encoder')
        decoder = keras.models.load_model('udacity-vae-decoder')

    vae = VAE(encoder, decoder, use_mse=USE_MSE)
    vae.compile(optimizer=keras.optimizers.Adam())

udacity_images = load_all_images(cfg)

i = 0
losses = []
for x in tqdm(udacity_images):
    i = i + 1

    x = utils.resize(x)
    x = normalize_and_reshape(x)

    loss = vae.test_on_batch(x)[1]  # total loss
    losses.append(loss)

    if i % 50 == 0:
        z_mean, z_log_var, z = encoder.predict(x)
        decoded = decoder.predict(z)

        reconstructed = vae.predict(x)
        plot_pictures_orig_rec(x, decoded, None, loss)

plt.figure(figsize=(20, 4))
x_losses = np.arange(len(losses))
plt.plot(x_losses, losses, color='blue', alpha=0.7)

plt.ylabel('Loss')
plt.xlabel('Number of Instances')
plt.title("Reconstruction error")

plt.show()

plt.clf()
plt.hist(losses, bins=len(losses) // 5)  # TODO: find an appropriate constant
plt.show()
