import tensorflow as tf
from tensorflow import keras, Tensor, Variable
from tensorflow.keras.layers import Layer
from tensorflow.data import Dataset
from decoders import get_simple_decoder
from typing import Optional
from tensorflow_probability import distributions as tfd


class ParticleAutoencoder(keras.Model):
    """A 'particle autoencoder' model to be trained with PGD (similar to in
    Sec. 3.3 of https://arxiv.org/abs/2204.12965).
    Args:
        latent_var_dim: Dimension of the latent space: integer.
        decoder: The decoder, or generator, network to be used in the model:
            a `Layer` object mapping from the latent space to the data
            space.
        prior: Prior distribution over latent space: a `tfd.Distribution`
            object. Optional: if left unspecified, it'll be set to an
            isotropic zero-mean unit-variance Gaussian.
        observation_noise_std: Standard deviation of observation noise: float.
    """

    # TODO: Add usage examples to docstring.
    def __init__(self,
                 latent_var_dim: int = 32,
                 decoder: Layer = None,
                 prior: Optional[tfd.Distribution] = None,
                 observation_noise_std: float = 1.):
        super().__init__()
        # TODO: Add type checks.
        # TODO: Add checks for dimensional compatibility of latent_var_dim,
        #  decoder, data, etc.
        self.latent_var_dim = latent_var_dim
        self._decoder = decoder()
        if prior is None:
            self._prior = tfd.MultivariateNormalDiag(
                loc=tf.zeros((self.latent_var_dim,)),
                scale_diag=tf.ones((self.latent_var_dim,)))
        else:
            self._prior = prior
        self._observation_noise_std = observation_noise_std
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

    def call(self, **kwargs):
        pass

    # TODO: Turn the above into a decoder call?

    def compile(self,
                lv_learning_rate: float = 1e-2,
                n_particles: int = 1,
                preprocessor: Layer = None,
                postprocessor: Layer = None,
                **kwargs):
        # Save latent variable learning rate, number of particles,
        # preprocessor, and postprocessor:
        self._lv_learning_rate = lv_learning_rate
        self._n_particles = n_particles
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        # Run normal compile:
        super().compile(**kwargs)

    def reset_particles(self, n_particles: Optional[int] = None) -> None:
        """Resets particles used in training by drawing samples from the
        prior.
        Args:
            n_particles: Number of particles: int. Optional: if left
            unspecified the number of particles will remain unchanged.
        """
        if n_particles:
            self._n_particles = n_particles
        self._train_latent_variables = tf.Variable(
            initial_value=self._prior.sample((self._n_particles,
                                              self._training_set_size)))

    def fit(self,
            data: Optional[Dataset] = None,
            batch_size: int = 64,
            shuffle_buffer_size: int = 1024,
            **kwargs):
        """Fits model to data.
        Args:
            data: dataset: `Dataset` object that yields batches of data with
                dimensions (batch_size, data_dims).
            batch_size: Batch size to be used in training: int.
            shuffle_buffer_size: Buffer size to be used for dataset shuffling:
                int.
        """
        self._training_set_size = len(data)
        # If particles uninitialized, initialize them:
        if self._train_latent_variables is None:
            self.reset_particles()

        # Adapt any pre-or-postprocessors:
        if self._preprocessor is not None:
            # If the preprocessor needs adapting, adapt it.
            if hasattr(self._preprocessor, 'adapt'):
                self._preprocessor.adapt(data)
        # If the postprocessor needs adapting, adapt it.
        if hasattr(self._postprocessor, 'adapt'):
            self._postprocessor.adapt(data)

        # Add indices to dataset and shuffle:
        data = Dataset.zip((Dataset.range(self._training_set_size), data))
        # Shuffle and batch data:
        self._train_batch_size = batch_size
        data = data.shuffle(shuffle_buffer_size)
        data = data.batch(self._train_batch_size, drop_remainder=True)

        # Declare variables to hold latent variables updated in each step.
        self._latent_var_batch = tf.Variable(initial_value=tf.zeros(
            shape=(self._n_particles * self._train_batch_size,
                   self.latent_var_dim)))
        # Run normal fit:
        return super().fit(x=data, **kwargs)
        # TODO: Delete latent variables at end of training to save memory?

    def train_step(self, data: Tensor) -> dict[str, float]:
        """Implements PGD training step.
        Args:
             data: (indices, data) batch yielded by `Dataset` object  passed
                into super().fit() at the end of self.fit: `Tensor` object.
        """
        # TODO: Rename loss in return and add description to docstring.
        # Unpack datapoints and corresponding indices:
        d_idx, data_batch = data
        d_idx = tf.repeat(d_idx, self._n_particles, axis=0)
        data_batch = tf.repeat(data_batch, self._n_particles, axis=0)
        # Extract latent variables to be updated:
        p_idx = tf.concat([tf.range(self._n_particles, dtype=tf.int64)
                           for _ in range(self._train_batch_size)], axis=0)
        lv_idx = tf.stack([p_idx, d_idx], axis=1)

        self._latent_var_batch.assign(
            self._train_latent_variables.gather_nd(lv_idx))

        with tf.GradientTape(persistent=True) as tape:
            # Compute log of model density:
            log_dens = tf.math.reduce_sum(self._log_density(data_batch,
                                                            self._latent_var_batch))
            # Scale it for parameter loss (negative sign so we take an ascent
            # step rather than a descent step in the optimizer):
            loss = - log_dens / (self._n_particles * self._train_batch_size)

        # Compute gradients:
        param_grads = tape.gradient(loss, self._decoder.trainable_variables)
        lv_grads = tape.gradient(log_dens, self._latent_var_batch)

        # Update parameters:
        self.optimizer.apply_gradients(zip(param_grads,
                                           self._decoder.trainable_variables))
        # Update latent variables:
        lv_lr = self._lv_learning_rate  # Get learning rate
        noise = tf.random.normal(shape=tf.shape(self._latent_var_batch))
        self._train_latent_variables.scatter_nd_add(lv_idx, lv_lr * lv_grads
                                                    + tf.sqrt(
            2 * lv_lr) * noise)

        return {'loss': loss}

    def _log_density(self, data: Tensor, latent_vars: Variable) -> Tensor:
        """Returns model's log density evaluated at each matching (data,
        latent-variable) pair.
        Args:
            data: Data batch (replicated across particles and flattened in the
                particle dimension): `Tensor` object with dimensions
                (batch_size * self._n_particles, data_dims).
            latent_vars: Latent variable batch (for each particle, flattened in
                the particle dimension): `Tensor` object with dimensions
                (batch_size * self._n_particles, self._latent_var_dim).
        Returns:
            Log density evaluations: `Tensor` object with dimensions
                (batch_size * n_particles).
        """
        # Preprocess data batch:
        if self._preprocessor is not None:
            data = self._preprocessor(data)
        # Compute log prior probability:
        log_dens = self._prior.log_prob(latent_vars)
        likelihood = tfd.MultivariateNormalDiag(loc=self._decoder(latent_vars),
                                                scale_diag=self._observation_noise_std * tf.ones_like(
                                                    data,
                                                    dtype=tf.float32))
        ll = likelihood.log_prob(data)
        log_dens += tf.math.reduce_sum(ll, axis=list(range(1, len(ll.shape))))
        return log_dens

    def decode(self, latent_vars: Tensor) -> Tensor:
        """Decodes samples. If any postprocessor was specified in the .compile
        call, the decoded samples are passed through it before being returned.
        Args:
            latent_vars: Batch of latent variables to be decoded: `Tensor`
                object of dimensions (batch_size, self._latent_var_dim).
        Returns:
            Decoded samples: `Tensor` object with dimensions
                (batch_size, data_dims).
        """
        if self._postprocessor is None:
            return self._decoder(latent_vars)
        else:
            return self._postprocessor(self._decoder(latent_vars))

    def encode(self, datapoint: Tensor) -> Tensor:
        pass

    def decode_posterior_samples(self,
                                 index: int = 0,
                                 n_samples: int = 1) -> Tensor:
        """Decodes latent variables corresponding to the specified index of
        n_sample many particles used in training.
        Args:
            index: Index of datapoint whose latent variables are to be decoded:
                int.
            n_samples: Number of particles to decode: int.
        Returns:
            Decoded particles: `Tensor` object with dimensions
                (n_samples, data_dims).
        """
        # We cannot return more samples than there are particles:
        n_samples = min(n_samples, self._n_particles)
        samples = self._train_latent_variables[:n_samples, index, :]
        return self.decode(samples)

    def generate_fakes(self, n_fakes: int = 1) -> Tensor:
        """
        Args:

        Returns:

        """
        return self.decode(self._prior.sample((n_fakes,)))
