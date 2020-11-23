import tensorflow as tf
from tensorflow.keras import Input, Model, losses
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.utils import get_custom_objects

from utils import RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS

original_dim = RESIZED_IMAGE_HEIGHT * RESIZED_IMAGE_WIDTH * IMAGE_CHANNELS


# custom loss function
def vae_loss(args):
    """
    Defines the VAE loss functions as a combination of MSE loss and KL-divergence loss.
    """
    x, x_decoded_mean, z_mean, z_log_sigma = args
    rec_loss = losses.mean_squared_error(x, x_decoded_mean)

    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)

    # return rec_loss + kl_loss
    # Total loss = 50 % rec + 50 % KL divergence loss
    return K.mean(rec_loss + kl_loss)


def sampling(args):
    """Re-parameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


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


class VariationalAutoencoder:

    def __init__(self,
                 model_name,
                 intermediate_dim=512,
                 latent_dim=2,
                 loss="VAE",  # [ "VAE", "MSE"]
                 learning_rate=1.0e-4):
        self.model_name = model_name
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.loss = loss
        self.learning_rate = learning_rate

    def create_autoencoder(self):
        intermediate_dim = self.intermediate_dim
        latent_dim = self.latent_dim
        input_shape = (original_dim,)

        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_sigma = Dense(latent_dim, name='z_log_sigma')(x)

        # use re-parameterization trick to push the sampling out as input
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_sigma])

        # instantiate the encoder model
        encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

        # create decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001))(
            latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate the decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')

        # instantiate the VAE model
        outputs = decoder(encoder(inputs)[2])
        # outputs = decoder(encoder(inputs))
        vae = Model(inputs, outputs, name='vae_mlp')

        loss = Lambda(vae_loss, name='vae_loss')([inputs, outputs, z_mean, z_log_sigma])

        # compile the model
        if "VAE" in self.loss:
            print("Using VAE loss")
            vae.compile(optimizer='adadelta')
            vae.add_loss(loss)
        elif "MSE" in self.loss:
            print("Using MSE loss")
            vae.compile(optimizer='adam', loss="mean_squared_error")
        else:
            print("Invalid loss value. Using default VAE loss.")
            vae.compile(optimizer='adadelta')
            vae.add_loss(loss)

        return vae
