import os
import shutil
import pickle
from typing import Optional
import tensorflow as tf
from tensorflow import keras, Tensor, Variable
from tensorflow.keras.layers import Layer
from tensorflow.data import Dataset
from tensorflow_probability import distributions as tfd
from sklearn.mixture import GaussianMixture
import numpy as np


#TODO: Add dimensionality checks for decoder vs data.
#TODO: Add examples to doctrings.
#TODO: Reshape internal particle representation to have dims
# (batch, n_part, lv_dims)
class LPAE(keras.Model):
    """
    A 'Langevin particle autoencoder' model grouping a prior distribution, a
    decoder, and a set of particles into a keras.Model-like object with
    training/inference features.

    Parameters
    ----------
    latent_dimensions: int
        Dimension of the latent space.
    decoder: tensorflow.keras.Layer
        The decoder, or generator, network mapping from the latent space to
        the data space.
    prior: tensorflow_probability.distributions.Distribution or None
        Prior distribution over latent space. Defaults to an isotropic
        zero-mean unit-variance Gaussian.
    observation_noise_std: float
        Standard deviation of observation noise. Defaults to one.

    Methods
    -------
    """
    def __init__(self,
                 latent_dimensions: int,
                 decoder: Layer,
                 prior: Optional[tfd.Distribution] = None,
                 observation_noise_std: float = 1e-2):
        super().__init__()
        # TODO: Add type checks.
        # TODO: Add checks for dimensional compatibility of _latent_dimensions,
        #  decoder, data, etc.
        self._latent_dimensions = latent_dimensions
        self._decoder = decoder
        if prior is None:
            self._default_prior = True
            self._prior = tfd.MultivariateNormalDiag(
                loc=tf.zeros((self._latent_dimensions,)),
                scale_diag=tf.ones((self._latent_dimensions,)))
        else:
            self._default_prior = False
            self._prior = prior
        self._observation_noise_std = observation_noise_std
        self._training_set_size = None  # Set when fit is called.
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
        self._gmm = None  # GMM approximation to the aggregate posterior, set
        # when generate_fakes is called.
        self._built = False  # True if model built, false otherwise.

    def _build(self, data: Dataset):
        """Builds model. That is, instantiates parameter and particle
        variables and adapts pre and post processors to data.

        Parameters
        ----------
        data: tensorflow.data import Dataset object
            Training dataset.
        """
        # Build decoder:
        self._decoder.build((self._latent_dimensions,))

        # If particles not built, build them:
        self._training_set_size = len(data)
        if self._train_latent_variables is None:
            self.reset_particles()

        # Build and adapt any pre-or-postprocessors:
        data_shape = data.element_spec.shape
        if self._preprocessor is not None:
            self._preprocessor.build(data_shape)
            # If the preprocessor needs adapting, adapt it.
            if hasattr(self._preprocessor, 'adapt'):
                self._preprocessor.adapt(data)
        if self._postprocessor is not None:
            self._postprocessor.build(data_shape)
            # If the postprocessor needs adapting, adapt it.
            if hasattr(self._postprocessor, 'adapt'):
                self._postprocessor.adapt(data)

    def call(self, data: Tensor, **kwargs):
        """
        Pass batch of data through the autoencoder (i.e., first encode and then
        decode it).

        Parameters
        ----------
        data: tensorflow.Tensor of dimensions (batch_size, data_dims)
            Batch of data.
        kwargs:
            Arguments to be passed on to self.encode.

        Returns
        -------
        tensorflow.Tensor of dimensions (batch_size, data_dims).
            Batch of encoded-and-then-decoded data
        """
        return self.decode(self.encode(data, **kwargs))

    def compile(self,
                lv_learning_rate: float = 1e-2,
                n_particles: int = 1,
                preprocessor: Optional[Layer] = None,
                postprocessor: Optional[Layer] = None,
                **kwargs):
        """
        Configures model for training.

        Parameters
        ----------
        lv_learning_rate: float
            Learning rate used in the latent variable updates. Defaults to
            0.01.
        n_particles: int
            Number of particles.
        optimizer: tensorflow.keras.optimizers.Optimizer
            Optimizer used to update model parameters.
        preprocessor: tensorflow.keras.layers.Layer or None
            Preprocessing layer to be applied to data prior to encoding. If the
            layer has an adapt method, the method will be called on the
            training set before training.
        postprocessor: tensorflow.keras.layers.Layer or None
            Postprocessing layer to be applied to decoded latent variables. If
            the layer has an adapt method, the method will be called on the
            training set before training.
        """
        # Save latent variable learning rate, number of particles,
        # preprocessor, and postprocessor:
        self._lv_learning_rate = lv_learning_rate
        self._n_particles = n_particles
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        # Run normal compile:
        super().compile(**kwargs)

    def reset_particles(self, n_particles: Optional[int] = None) -> None:
        """
        Resets particles used in training by drawing samples from the prior.

        Parameters
        ----------
        n_particles: int or None
            Number of particles. If None, the number of particles will remain
            unchanged.
        """
        if n_particles:
            self._n_particles = n_particles
        self._train_latent_variables = tf.Variable(
            initial_value=self._prior.sample((self._n_particles,
                                              self._training_set_size)))

    #TODO: Add description of training loss numbers.
    def fit(self,
            data: Dataset,
            batch_size: int = 64,
            shuffle_buffer_size: int = 1024,
            **kwargs):
        """
        Trains model using the subsampled version of PGD described in the docs.

        Parameters
        ----------
        data: tensorflow.dataset.Dataset object that yields batches of data with dimensions (batch_size, data_dims).
            Training set.
        batch_size: int
            Batch size to be used in training. Defaults to 64.
        shuffle_buffer_size: int
            Buffer size to be used for dataset shuffling. Defaults to 1024.

        Returns
        -------
        A tensorflow History object.
            Its History.history attribute is a record of training loss values.
        """
        # Build all parameters and particles, adapt pre/postprocessors:
        self._build(data)

        # Add indices to dataset and shuffle:
        data = Dataset.zip((Dataset.range(self._training_set_size), data))
        # Shuffle and batch data:
        self._train_batch_size = batch_size
        data = data.shuffle(shuffle_buffer_size)
        data = data.batch(self._train_batch_size, drop_remainder=True)

        # Declare variables to hold latent variables updated in each step.
        self._latent_var_batch = tf.Variable(initial_value=tf.zeros(
            shape=(self._n_particles * self._train_batch_size,
                   self._latent_dimensions)))
        # Run normal fit:
        return super().fit(x=data, **kwargs)
        # TODO: Delete latent variables at end of training to save memory?

    def train_step(self, data: Tensor) -> dict[str, float]:
        """
        Implements PGD training step.

        Parameters
        ----------
         data: tensorflow.Tensor
            (indices, data) batch yielded by `Dataset` object  passed
            into super().fit() at the end of self.fit.
        """
        # TODO: Rename loss in return and add description to docstring.
        # TODO: Can we speed the following using tf.gather like in
        #  self.get_particles (together with tf.reshape to flatten)?
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
            log_dens = tf.math.reduce_sum(self._log_density(
                data_batch, self._latent_var_batch))
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

    # TODO: Add test step to monitor validation losses?

    def _log_density(self, data: Tensor, latent_vars: Variable) -> Tensor:
        """
        Returns model's log density evaluated at each matching (data,
        latent-variable) pair.

        Parameters
        -----------
        data: tensorflow.Tensor with dimensions (batch_size
        * self._n_particles, data_dims)
            Data batch (replicated across particles and flattened in the
            particle dimension).
        latent_vars: tensorflow.Variable with dimensions (batch_size
        * self._n_particles, self._latent_var_dim).
            Latent variable batch (for all particles, flattened in the particle
            dimension).


        Returns
        --------
        tensorflow.Tensor with dimensions (batch_size * n_particles).
            Tensor of log density evaluations.
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
        """
        Decodes batch of latent variables. If any postprocessor was specified
        in the .compile call, the decoded samples are passed through it before
        being returned.

        Parameters
        -----------
        latent_vars: tensorflow.Tensor of dimensions (batch_size, n_particles,
        latent_dim).
             Batch of latent variables to be decoded

        Returns
        --------
        tensorflow.Tensor with dimensions (batch_size, n_particles, data_dims)
            Batch of decoded latent variables.
        """
        batch_size, n_particles, _ = latent_vars.shape
        decoded = self._decoder(self.flatten_particles(latent_vars))
        if self._postprocessor is not None:
            decoded = self._postprocessor(decoded)
        return tf.reshape(decoded,
                          (batch_size, n_particles) + decoded.shape[1:])

    # TODO: Test encode and decode with multiple particles.
    def encode(self, data: Tensor,
               n_steps: int = 100,
               step_size: float = 1e-5,
               n_particles: int = 1) -> Tensor:
        """
        Encodes batch of data by running a ULA chain per datapoint in batch.

        Parameters
        -----------
        data: tensorflow.Tensor of dimensions (batch_size, data_dims)
            Batch of data.
        n_steps: int
            Number of ULA steps to taken to encode. Defaults to 1000.
        step_size: int
            ULA step size used to encode. Defaults to 1e-5.
        n_particles: int
            Number of particles to generate. Defaults to 1.

        Returns
        --------
        tensorflow.Tensor with dimensions (batch_size, n_particles,
        latent dimensions)
            Batch of latent variables.
        """
        # Freeze model parameters:
        self.trainable = False

        # Declare latent variables:
        latent_vars = tf.Variable(initial_value
                                  =self._prior.sample((len(data) * n_particles,
                                                       )))
        # Repeat data across particles:
        rdata = tf.repeat(data, n_particles, axis=0)
        # Infer latent variables:
        inference_step = tf.function(self._inference_step
                                     ).get_concrete_function(rdata,
                                                             latent_vars,
                                                             self._log_density,
                                                             step_size)
        for _ in range(n_steps):
            inference_step(rdata, latent_vars)

        # Unfreeze model parameters:
        self.trainable = True

        return tf.reshape(latent_vars.read_value(), (data.shape[0],
                                                     n_particles,
                                                     self._latent_dimensions))

    @staticmethod
    def _inference_step(data: Tensor,
                        latent_vars: Variable,
                        log_density,
                        step_size: float) -> Tensor:
        with tf.GradientTape() as tape_inf:
            log_density_eval = tf.math.reduce_sum(log_density(data,
                                                              latent_vars))
        grads = tape_inf.gradient(log_density_eval, latent_vars)
        noise = tf.random.normal(shape=tf.shape(latent_vars))
        latent_vars.assign_add(step_size * grads
                               + tf.sqrt(2 * step_size) * noise)
        return log_density_eval

    def get_particles(self,
                      index_batch=None,
                      n_particles: Optional[int] = None) -> Tensor:
        """
        Gets particles for training datapoints indexed by indices in
        index_batch.

        Parameters
        -----------
        index_batch: integer containing array-like object with dimensions
        (batch_size) or None
            Batch of indices whose particles are to be fetched. If None,
            particles are fetched for all indices.
        n_particles: int or None
            Number of particles to fetch per index. If None, all particles are
            fetched.

        Returns
        -------
        tensorflow.Tensor with dimensions (batch_size, n_particles,
        latent_var_dims)
            Fetched particles.
        """

        # If index_batch not specified, set it to be that of all indices:
        if index_batch is None:
            index_batch = tf.range(self._training_set_size, dtype=tf.int64)
        # If n_particles not specified, set it to be the number of particles:
        if n_particles is None:
            n_particles = self._n_particles

        # Make sure index_batch is a tensor with int64 dtype:
        if not tf.is_tensor(index_batch):
            index_batch = tf.convert_to_tensor(index_batch, dtype=tf.int64)
        if not (index_batch.dtype == tf.int64):
            index_batch = tf.cast(index_batch, tf.int64)

        # Get and return particles:
        return tf.transpose(tf.gather(self._train_latent_variables,
                                      index_batch, axis=1)[:n_particles, ...],
                            [1, 0, 2])

    @staticmethod
    def flatten_particles(particles: Tensor) -> Tensor:
        """
        Flattens the particle number and batch size dimensions of the particles
        so that they can be efficiently fed into the decoder.

        Parameters
        -----------
        particles: Tensor with dimensions (n_particles, batch_size,
        latent_dimension)
            Particle to be flattened.

        Returns
        -------
        Flattened particles.
        """

        return tf.reshape(particles, [particles.shape[0] * particles.shape[1],
                                      particles.shape[2]])

    def decode_particles(self, index_batch, n_particles: int = None) -> Tensor:
        """
        Decodes n_particles particles for each training datapoint indexed by
        index_batch.

        Parameters
        -----------
        index_batch: integer containing array-like object with dimensions
        (batch_size)
            Batch of indices of datapoints whose particles are to be fetched
            and decoded.
        n_particles: int or None
            Number of particles to decode per index. If None, all particles are
            decoded.

        Returns
        -------
        tensorflow.Tensor with dimensions (batch_size, n_particles, data_dims)
            Decoded particles.
        """

        return self.decode(self.get_particles(index_batch, n_particles))

    def generate_fakes(self,
                       n_fakes: int = 1,
                       from_prior: bool = False,
                       n_components: Optional[int] = None) -> Tensor:
        """
        Generates batch of fake datapoints by sampling latent variables and
        mapping them throw the decoder.

        Parameters
        -----------
        n_fakes: int
            Number of fake datapoints.
        from_prior: bool
            If set to true, the latent variables are drawn from
            the prior. Otherwise, the aggregate posterior is approximated
            using a mixture of Gaussians (MoG) and the latent variables are
            drawn from the mixture. Computing the MoG usually takes some time,
            but once computed it's saved for later use.
        n_components: int
            Number of components in mixture. Defaults to 30.

        Parameters
        -----------
        tensorflow.Tensor object with dimensions (n_fakes, data_dims)
            Batch of fake data points.
        """
        if from_prior:
            return self.decode(self._prior.sample((n_fakes, 1)))[:, 0, ...]

        # If necessary, fit gmm:
        if (not self._gmm) | bool(n_components):
            if not n_components:
                n_components = 30
            self._gmm = GaussianMixture(n_components=n_components)
            n, m = self._n_particles, self._training_set_size
            d = self._latent_dimensions
            self._gmm.fit(tf.reshape(self._train_latent_variables,
                                     [n * m, d]).numpy())

        # Draw latent variables, decode, and return:
        lvs, _ = self._gmm.sample(n_samples=n_fakes)
        return self.decode(lvs[:, tf.newaxis, :])[:, 0, ...]

    def get_config(self):
        config = {
            'decoder_class': self._decoder.__class__,
            '_latent_dimensions': self._latent_dimensions,
            '_observation_noise_std': self._observation_noise_std,
            '_training_set_size': self._training_set_size,
            '_n_particles': self._n_particles,
            '_lv_learning_rate': self._lv_learning_rate,
            '_train_batch_size': self._train_batch_size,
        }
        if self._preprocessor is not None:
            config['preprocessor_class'] = self._preprocessor.__class__
        if self._postprocessor is not None:
            config['postprocessor_class'] = self._postprocessor.__class__
        return config

    def save(self, path: str):
        """
        Saves model.

        Parameters
        ----------
        path: str
            Save directory.
            """
        if not self._default_prior:
            raise NotImplementedError('Save method does not support '
                                      'custom priors.')
        if self._decoder.__class__.__name__ not in {'Model', 'Sequential'}:
            raise NotImplementedError('Save method only supports decoders that'
                                      'are keras `Sequential` or `Model` '
                                      'objects.')
        # Create save directory (delete old if it exists):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        os.makedirs(path + 'decoder/')
        # Save config:
        config = self.get_config()
        with open(path + 'config.pkl', 'w+b') as f:
            pickle.dump(config, f)
        # Save decoder:
        with open(path + 'decoder/config.pkl', 'w+b') as f:
            pickle.dump(self._decoder.get_config(), f)
        self._decoder.save_weights(path + 'decoder/ckpt')
        # Save latent variables:
        np.save(path + 'train_latent_variables.npy',
                self._train_latent_variables)
        # If they exist, save preprocessor, postprocessor, and gmm:
        if self._preprocessor is not None:
            with open(path + 'preprocessor_config.pkl', 'w+b') as f:
                pickle.dump(self._preprocessor.get_config(), f)
        if self._postprocessor is not None:
            with open(path + 'postprocessor_config.pkl', 'w+b') as f:
                pickle.dump(self._postprocessor.get_config(), f)
        if self._gmm is not None:
            with open(path + 'gmm.pkl', 'w+b') as f:
                pickle.dump(self._gmm, f)

    @classmethod
    def from_save(cls, path: str):
        """
        Loads model.

        Parameters
        ---------
        path: str
            Save directory

        Returns
        -------
        LPAE object
            Loaded model.
        """
        # Load config:
        with open(path + 'config.pkl', 'rb') as f:
            config = pickle.load(f)

        # Instantiate decoder:
        with open(path + 'decoder/config.pkl', 'rb') as f:
            decoder_config = pickle.load(f)
        decoder = config['decoder_class'].from_config(decoder_config)
        decoder.load_weights(path + 'decoder/ckpt')

        # Instantiate LangevinAutoencoder object:
        lpae = cls(latent_dimensions=config['_latent_dimensions'],
                   decoder=decoder,
                   observation_noise_std=config['_observation_noise_std'])

        # Load latent variables:
        lpae._train_latent_variables = tf.Variable(
            np.load(path + 'train_latent_variables.npy'))

        # If they exist, load preprocessor, postprocessor, and gmm:
        if os.path.exists(path + 'preprocessor_config.pkl'):
            with open(path + 'preprocessor_config.pkl', 'rb') as f:
                pre_cfg = pickle.load(f)
            preprocessor = config['preprocessor_class'].from_config(pre_cfg)
        if os.path.exists(path + 'postprocessor_config.pkl'):
            with open(path + 'postprocessor_config.pkl', 'rb') as f:
                post_cfg = pickle.load(f)
            postprocessor = config['postprocessor_class'].from_config(post_cfg)
        if os.path.exists(path + 'gmm.pkl'):
            with open(path + 'gmm.pkl', 'rb') as f:
                lpae._gmm = pickle.load(f)

        # Compile object:
        lpae.compile(lv_learning_rate=config['_lv_learning_rate'],
                     n_particles=config['_n_particles'],
                     preprocessor=preprocessor,
                     postprocessor=postprocessor)

        # Load remaining attributes:
        lpae._training_set_size = config['_training_set_size']
        lpae._train_batch_size = config['_train_batch_size']

        return lpae
