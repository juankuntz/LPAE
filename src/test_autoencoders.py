import tensorflow as tf
from tensorflow.keras import Sequential
from .autoencoders import LPAE
from tensorflow.keras.layers import Conv2DTranspose, Dense, Reshape, LeakyReLU, \
    Conv2D
from tensorflow_probability import distributions as tfd
from tensorflow import TensorShape


def get_lpae_model():
    """LPAE model applicable to MNIST data."""
    decoder = Sequential([Dense(7 * 7 * 64, activation="relu",
                                input_shape=(32,)),
                          Reshape((7, 7, 64)),
                          Conv2DTranspose(64, 3, activation="relu", strides=2,
                                          padding="same"),
                          Conv2DTranspose(32, 3, activation="relu", strides=2,
                                          padding="same"),
                          Conv2DTranspose(1, 3, activation="sigmoid",
                                          padding="same")
                          ])
    return LPAE(latent_dimensions=32, decoder=decoder, observation_noise_std=1)


def get_compiled_lpae_model():
    model = get_lpae_model()
    model.compile(lv_learning_rate=1e-2, n_particles=10,
                  optimizer=tf.keras.optimizers.RMSprop(),
                  preprocessor=tf.keras.layers.Rescaling(scale=1. / 255),
                  postprocessor=tf.keras.layers.Rescaling(scale=255.))
    return model


def test_log_density_shape():
    model = get_compiled_lpae_model()
    data = tf.zeros([5, 28, 28, 1])
    latent_vars = tf.Variable(tf.zeros([5, model._latent_dimensions]))
    log_density = model._log_density(data, latent_vars)
    assert log_density.shape == (5,)


def test_log_density_computation():
    model = get_compiled_lpae_model()
    data = tf.zeros([5, 28, 28, 1])
    latent_vars = tf.Variable(tf.zeros([5, model._latent_dimensions]))
    log_density = model._log_density(data, latent_vars)
    likelihood = tfd.MultivariateNormalDiag(loc=model._decoder(latent_vars),
                                            scale_diag
                                            =model._observation_noise_std
                                             * tf.ones_like(data,
                                                            dtype=tf.float32))
    ll = likelihood.log_prob(data)
    ll = tf.math.reduce_sum(ll, axis=list(range(1, len(ll.shape))))
    assert (log_density
            == ll + model._prior.log_prob(latent_vars)).numpy().all()


def test_default_prior_shape():
    model = get_compiled_lpae_model()
    assert model._prior.event_shape == (model._latent_dimensions,)


def test_decode_shape():
    model = get_compiled_lpae_model()
    assert model.decode(tf.zeros((3, 2, 32))).shape == (3, 2, 28, 28, 1)


def test_encode_shape_1():
    model = get_compiled_lpae_model()
    assert model.encode(tf.zeros((3, 28, 28, 1))).shape == (3, 1, 32)


def test_encode_shape_2():
    model = get_compiled_lpae_model()
    assert (model.encode(tf.zeros((3, 28, 28, 1)), n_particles=5).shape
            == (3, 5, 32))