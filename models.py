import tensorflow as tf
from tensorflow import keras, Tensor, Variable
from tensorflow.keras.layers import Layer
from tensorflow.data import Dataset
from decoders import get_simple_decoder
from typing import Optional
from tensorflow_probability import distributions as tfd


class LAE(keras.Model):

    def __init__(self,
                 latent_var_dim: int = 32,
                 decoder: Optional[Layer] = None,
                 prior: Optional[tfd.Distribution] = None,
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
        self._train_latent_variables = None  # All latent variables. Built when
        # fit is called. Should be of dims (n_particles, train_size,
        # latent_var_dims).
        self._latent_var_batch = None  # Latent variables used in update. Built
        # when fit is called. Should be of dims (n_particles * batch_size,
        # latent_var_dims).

    def compile(self,
                lv_learning_rate: float = 1e-2,
                n_particles: int = 1,
                preprocessor: Layer = None,
                postprocessor: Layer = None,
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
            data: Dataset | None = None,
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
        data = data.shuffle(shuffle_buffer_size)
        data = data.batch(self._n_particles * self._train_batch_size,
                          drop_remainder=True)
        # TODO: Right now particles are shuffled: we don't update 0, ..., N-1
        # once every step.

        # Declare variables to hold latent variables updated in each step.
        self._latent_var_batch = tf.Variable(initial_value=tf.zeros(
            shape=(
                self._n_particles * self._train_batch_size,
                self.latent_var_dim)))
        # Run normal fit:
        return super().fit(x=data, **kwargs)
        # TODO: Delete latent variables at end of training to save memory?

    @staticmethod
    def _add_indices_to_dataset(data: Dataset, np: int):
        """Add particle and data indices to dataset. In particular, if data
        equals [A, B, C] and there are two particles, data is set to
        [[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2], [A, B, C, A, B, C]];
        that is [particle_indices, data_indices, datapoints].
        Args:
            - data: dataset to which we need to add indices.
            - np: number of particles.
        Returns: data set with indices."""
        tss = data.cardinality().numpy()
        return Dataset.zip((Dataset.range(np).map(lambda x: tf.repeat(x, tss)
                                                  ).flat_map(
            lambda y: Dataset.from_tensor_slices(y)),
                            Dataset.range(tss).repeat(np), data.repeat(np)))

    def train_step(self, data: Tensor):
        """Note that data is the training batch yielded by the data.dataset
        object passed into super().fit() at the end of self.fit."""
        # Unpack datapoints and corresponding indices:
        p_idx, d_idx, data_batch = data
        # Extract latent variables to be updated:
        lv_idx = tf.stack([p_idx, d_idx], axis=1)

        with tf.GradientTape(persistent=True) as tape:
            # Compute log of model density:
            log_dens = tf.math.reduce_sum(self._log_density(data_batch,
                                                            self._latent_var_batch))
            # Scale it for parameter loss (negative sign so we take an ascent
            # step rather than a descent step in the optimizer):
            loss = - log_dens / (self._n_particles * self._train_batch_size)

        # Compute gradients:
        param_grads = tape.gradient(loss, self.decoder.trainable_variables)
        lv_grads = tape.gradient(log_dens, self._latent_var_batch)

        # Update parameters:
        self.optimizer.apply_gradients(zip(param_grads,
                                           self.decoder.trainable_variables))
        # Update latent variables:
        lv_lr = self._lv_learning_rate  # Get learning rate
        noise = tf.random.normal(shape=tf.shape(self._latent_var_batch))
        self._train_latent_variables.scatter_nd_add(lv_idx, lv_lr * lv_grads
                                                    + tf.sqrt(2 * lv_lr) * noise)

        return {'loss': loss}


    def _log_density(self, data: Tensor, latent_vars: Variable) -> Tensor:
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
        likelihood = tfd.MultivariateNormalDiag(loc=self.decode(latent_vars),
                                                scale_diag=tf.ones_like(data,
                                                                        dtype=tf.float32))
        ll = likelihood.log_prob(data)
        log_dens += tf.math.reduce_sum(ll, axis=list(range(1, len(ll.shape))))
        return log_dens

    def decode(self, latent_vars: Tensor) -> Tensor:
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
