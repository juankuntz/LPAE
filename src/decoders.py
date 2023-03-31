from tensorflow.keras.layers import Conv2DTranspose, Dense, Reshape
from tensorflow.keras import Sequential, Input


def get_simple_decoder(latent_var_dim: int):
    """Simple convolutional decoder for 28 x 28 single-channel images taken
    from https://keras.io/examples/generative/vae/."""
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