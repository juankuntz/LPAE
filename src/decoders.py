from tensorflow.keras.layers import (Conv2DTranspose, Dense, Reshape,
                                     LeakyReLU, Conv2D)
from tensorflow.keras import Sequential, Input


def get_simple_mnist_decoder(latent_var_dim: int):
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


def get_simple_celeba_decoder(latent_dim: int):
    """Simple convolutional decoder for 64 x 64 colour images taken
    from https://keras.io/examples/generative/dcgan_overriding_train_step/."""
    return Sequential(
        [
            Input(shape=(latent_dim,)),
            Dense(8 * 8 * 128),
            Reshape((8, 8, 128)),
            Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            LeakyReLU(alpha=0.2),
            Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name="decoder",
    )
