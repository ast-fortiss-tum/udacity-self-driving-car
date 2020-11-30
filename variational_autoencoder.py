import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

from utils import RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS

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
    def __init__(self, model_name, loss, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.model_name = model_name
        self.intermediate_dim = 512
        self.latent_dim = 2
        self.loss = loss,
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(keras.losses.mean_squared_error(data, reconstruction))
            reconstruction_loss *= RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT

            if "VAE" in self.loss:
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
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(keras.losses.mean_squared_error(inputs, reconstruction))
        reconstruction_loss *= RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT

        if "VAE" in self.loss:
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


def get_input_shape():
    return (original_dim,)


def get_image_dim():
    return RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS


def normalize_and_reshape(x):
    x = x.astype('float32') / 255.
    x = x.reshape(-1, RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS)
    return x


def reshape(x):
    x = x.reshape(-1, RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS)
    return x

# class VariationalAutoencoder:
#
#     def __init__(self,
#                  model_name,
#                  intermediate_dim=512,
#                  latent_dim=2,
#                  loss="VAE"):  # [ "VAE", "MSE"]
#         self.model_name = model_name
#         self.intermediate_dim = intermediate_dim
#         self.latent_dim = latent_dim
#         self.loss = loss
#
#     def create_autoencoder(self):
#         intermediate_dim = self.intermediate_dim
#         latent_dim = self.latent_dim
#         input_shape = (original_dim,)
#
#         # build encoder model
#         inputs = Input(shape=input_shape, name='encoder_input')
#         x = Dense(intermediate_dim, activation='relu')(inputs)
#         z_mean = Dense(latent_dim, name='z_mean')(x)
#         z_log_sigma = Dense(latent_dim, name='z_log_sigma')(x)
#
#         # use re-parameterization trick to push the sampling out as input
#         z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_sigma])
#
#         # instantiate the encoder model
#         encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
#
#         # create decoder model
#         latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
#         x = Dense(intermediate_dim,
#                   activation='relu',
#                   kernel_regularizer=tf.keras.regularizers.l1(0.001))(latent_inputs)
#         outputs = Dense(original_dim, activation='sigmoid')(x)
#
#         # instantiate the decoder model
#         decoder = Model(latent_inputs, outputs, name='decoder')
#
#         # instantiate the VAE model
#         outputs = decoder(encoder(inputs)[2])
#         # outputs = decoder(encoder(inputs))
#         vae = Model(inputs, outputs, name='vae_mlp')
#
#         loss = Lambda(vae_loss, name='vae_loss')([inputs, outputs, z_mean, z_log_sigma])
#
#         # compile the model
#         if "VAE" in self.loss:
#             print("Using VAE loss")
#             vae.compile(optimizer='adadelta')
#             vae.add_loss(loss)
#         elif "MSE" in self.loss:
#             print("Using MSE loss")
#             vae.compile(optimizer='adam', loss="mean_squared_error")
#         else:
#             print("Invalid loss value. Using default VAE loss.")
#             vae.compile(optimizer='adadelta')
#             vae.add_loss(loss)
#
#         return vae, encoder, decoder
