from tensorflow.keras.layers import (Conv2D, Conv2DTranspose,
                                     BatchNormalization, Dense, Reshape)
from tensorflow.keras import Input, Sequential, Model
from tensorflow import keras


def get_simple_decoder(latent_var_dim: int):
    """Simple convolutional decoder for 28 x 28 single-chanel images taken from
    https://keras.io/examples/generative/vae/."""
    return Sequential([Dense(7 * 7 * 64, activation="relu",
                             input_shape=(latent_var_dim,)),
                       Reshape((7, 7, 64)),
                       Conv2DTranspose(64, 3, activation="relu", strides=2,
                                       padding="same"),
                       Conv2DTranspose(32, 3, activation="relu", strides=2,
                                       padding="same"),
                       Conv2DTranspose(1, 3, activation="sigmoid",
                                       padding="same")
                       ], name='decoder')


def get_parem_generator():
    """Returns the generator network in
    https://github.com/juankuntz/ParEM/blob/main/torch/parem/models.py."""
    pass


def get_deterministic_layer(n_channels_in: int,
                            n_channels_out: int,
                            activation=keras.activations.gelu):
    """Returns a deterministic layer for the parem generator."""
    input = Input(shape=(None, None))
    x = Conv2D(filters=n_channels_out, kernel_size=5, padding='same')(input)
    x = BatchNormalization()(x)
    x = activation(x)
    x = Conv2D(filters=n_channels_out, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = activation(x)
    output = x + input  # Skip connection
    return Model(input, output)


def get_projection_layer(coef: int = 4, ngf: int = 16,
                         activation=keras.activations.gelu):
    """Returns a projection layer for the parem generator."""

    return Sequential([Dense(coef * ngf * ngf)(input),
                       BatchNormalization(),
                       activation,
                       Conv2DTranspose(ngf * coef, kernel_size=5,
                                       padding='same',
                                       use_bias=False),
                       BatchNormalization(),
                       activation])
