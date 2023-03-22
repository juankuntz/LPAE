import tensorflow as tf
from tensorflow import keras
from decoders import get_simple_decoder
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt


class LAE(keras.Model):

    def __init__(self,
                 latent_var_dim: int = 32,
                 decoder: keras.layers.Layer | None = None,
                 prior: tfd.Distribution | None = None,
                 observation_noise: float = 1.):
        super().__init__()
        # TODO: Current decoder only allows for (28, 28) single channel images
        # figure out how best to generalize this.
        # TODO: Add type checks.
        self.latent_var_dim = latent_var_dim
        if decoder is None:
            self.decoder = get_simple_decoder(self.latent_var_dim)
        else:
            self.decoder = decoder()
        if prior is None:
            self._prior = tfd.MultivariateNormalDiag(
                loc=tf.zeros((self.latent_var_dim,)),
                scale_diag=tf.ones((self.latent_var_dim,)))
        else:
            self._prior = prior
        self.observation_noise = observation_noise
        self._training_set_size = None  # Set weh fit is called.
        self._n_particles = None  # Set when compile is called.
        self._lv_learning_rate = None  # Set when compile is called.
        self._preprocessor = None  # Set when compile is called.
        self._postprocessor = None  # Set when compile is called.
        self._train_batch_size = None  # Set when fit is called.
        self._train_latent_variables = None  # Built when fit is called. Should
        # be of dims (n_particles, train_size, data_dims)

    def compile(self,
                lv_learning_rate: float = 1e-2,
                n_particles: int = 1,
                preprocessor: tf.keras.layers.Layer = None,
                postprocessor: tf.keras.layers.Layer = None,
                **kwargs):
        # Save latent variable learning rate, number of particles, and
        # preprocessor:
        self._lv_learning_rate = lv_learning_rate
        self._n_particles = n_particles
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        # Run normal compile:
        super().compile(**kwargs)

    def fit(self,
            data: tf.data.Dataset | None = None,
            reset_particles: bool = False,
            batch_size: int = 64,
            shuffle_buffer_size: int = 1024,
            **kwargs):
        """data must yield batches of data of dimensions
        (batch_size, data_dims)."""
        # If particles uninitialized, or set to be reset, initialize them:
        if (self._train_latent_variables is None) | reset_particles:
            self._training_set_size = len(data)
            self._train_latent_variables = tf.Variable(
                initial_value=self._prior.sample((self._n_particles,
                                                  self._training_set_size)))
            # trainable=True)

        # Adapt any pre-or-postprocessors:
        if self._preprocessor is not None:
            # If the preprocessor needs adapting, adapt it.
            if hasattr(self._preprocessor, 'adapt'):
                self._preprocessor.adapt(data)
        # If the postprocessor needs adapting, adapt it.
        if hasattr(self._postprocessor, 'adapt'):
            self._postprocessor.adapt(data)

        # Add particle and data indices to dataset:
        data = self._add_indices_to_dataset(data, self._n_particles)
        # Shuffle and batch data:
        self._train_batch_size = batch_size
        data = data.shuffle(shuffle_buffer_size).batch(self._train_batch_size)

        # Run normal fit:
        return super().fit(x=data, **kwargs)
        # TODO: Delete latent variables at end of training to save memory?

    @staticmethod
    def _add_indices_to_dataset(data: tf.data.Dataset, np: int):
        """Add particle and data indices to dataset. In particular, if data
        equals [A, B, C] and there are two particles, data is set to
        [[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [A, B, C, A, B, C]];
        that is [particle_indices, data_indices, datapoints].
        Args:
            - data: dataset to which we need to add indices.
            - np: number of particles.
        Returns: data set with indices."""
        tss = data.cardinality().numpy()
        return tf.data.Dataset.zip((
            tf.data.Dataset.range(np).map(
                lambda x: tf.repeat(x, tss)).flat_map(
                lambda y: tf.data.Dataset.from_tensor_slices(y)),
            tf.data.Dataset.range(tss).repeat(np),
            data.repeat(np)))

    def train_step(self, data):
        """Note that data is the training batch yielded by the data.dataset
        object passed into super().fit() at the end of self.fit."""
        # Unpack datapoints and corresponding indices:
        p_idx, d_idx, data_batch = data
        # Extract latent variables to be updated:
        lv_idx = tf.stack([p_idx, d_idx], axis=1)
        latent_var_batch = tf.Variable(
            initial_value=self._train_latent_variables.gather_nd(lv_idx))

        # Compute log density:
        with tf.GradientTape(persistent=True) as tape:
            # Compute log of model density:
            log_dens = tf.math.reduce_sum(self._log_density(data_batch,
                                                            latent_var_batch))
            # Scale it for parameter loss (negative sign so we take an ascent
            # step rather than a descent step in the optimizer):
            loss = - log_dens / (self._n_particles * self._train_batch_size)
            # self._training_set_size *

        # Compute gradients:
        dec_params = self.decoder.trainable_variables
        dec_grads = tape.gradient(loss, dec_params)
        lv_grads = tape.gradient(log_dens, latent_var_batch)
        # Take a step:
        self.optimizer.apply_gradients(zip(dec_grads, dec_params))
        lv_lr = self._lv_learning_rate
        self._train_latent_variables.scatter_nd_add(lv_idx, (
                lv_lr * lv_grads + tf.sqrt(2 * lv_lr) * tf.random.normal(
            shape=latent_var_batch.shape)))
        return {'loss': loss}
#abs(self._train_latent_variables).numpy().max()
    def _log_density(self, data: tf.Tensor,
                     latent_vars: tf.Variable) -> tf.Tensor:
        """Returns model's log density evaluated at each matching (data,
        latent-variable) pair.
        Args:
            - data: tensor of dimensions (batch_size * n_particles, data_dims).
            - latent_vars: tensor of dimensions (batch_size * n_particles,
            self.latent_var_dim).
        Returns: tensor of dimensions (batch_size * n_particles)."""
        # Normalize data batch:
        if self._preprocessor is not None:
            data = self._preprocessor(data)
        # Compute log prior probability:
        log_dens = self._prior.log_prob(latent_vars)
        likelihood = tfd.MultivariateNormalDiag(
            loc=self.decode(latent_vars),
            scale_diag=1e-2 ** 2 * tf.ones_like(data, dtype=tf.float32)
        )
        ll = likelihood.log_prob(data)
        log_dens += tf.math.reduce_sum(ll, axis=list(range(1, ll.ndim)))
        return log_dens

    def decode(self, latent_vars: tf.Tensor) -> tf.Tensor:
        """Returns decoded samples.
        Args:
            - latent_vars: tensor of dimensions (batch_size,
            self.latent_var_dim).
        Returns: tensor of dimensions (batch_size, data_dims)."""
        return self.decoder(latent_vars)

    def decode_posterior_samples(self, index: int = 0, n_samples: int = 1):
        n_samples = min(n_samples, self._n_particles)
        samples = self._train_latent_variables[:n_samples, index, :]
        decoded_samples = self.decoder(samples)
        if self._postprocessor is None:
            return decoded_samples
        else:
            return self._postprocessor(decoded_samples)
