from tensorflow.keras.layers import Conv2D, Conv2DTranspose, \
    BatchNormalization, Dense, Reshape
from tensorflow.keras import Input, Sequential


def get_simple_decoder(latent_var_dim: int):
    """Simple convolutional decoder for 28 x 28 single-chanel images taken from
    https://keras.io/examples/generative/vae/."""
    return Sequential([Dense(7 * 7 * 64, activation="relu",
                             input_shape=(latent_var_dim, )),
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


def get_deterministic_layer(activation_type: str = 'relu'):
    """Returns a deterministic layer for the parem generator."""
    pass


def get_projection_layer():
    """Returns a projection layer for the parem generator."""
    pass
