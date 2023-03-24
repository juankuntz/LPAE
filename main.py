from datetime import datetime
import models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow_probability import distributions as tfd


# Load data:
(x, y), (_, _) = tf.keras.datasets.mnist.load_data()
x = x[:10000].astype('float32')[..., np.newaxis]
data = tf.data.Dataset.from_tensor_slices(x)

lv_dim = 2
prior = tfd.MultivariateNormalDiag(loc=tf.zeros((lv_dim,)),
                                   scale_diag=tf.ones((lv_dim,)))
# Setup model:
lr = 1e-3/2
lae = models.LAE(latent_var_dim=2, prior=prior, observation_noise_std=1e-2)
lae.compile(lv_learning_rate=lr, n_particles=2,
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr),
            preprocessor=tf.keras.layers.Rescaling(scale=1./255),
            postprocessor=tf.keras.layers.Rescaling(scale=255.))
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = lae.fit(data=data, epochs=7, batch_size=64, callbacks=[tb])

i = 0
samples = lae.decode_posterior_samples(n_samples=3, index=i)
images = [x[i, ..., 0]] + [samples[i, ...].numpy()[..., 0] for i in range(samples.shape[0])]
grid_size = math.ceil(len(images) ** (1/2))
for i, image in enumerate(images):
    ax = plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(255 - image.astype("uint8"), cmap='Greys')
    plt.axis("off")
plt.show()

a=1
