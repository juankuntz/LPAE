from datetime import datetime
import models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

(x, y), (_, _) = tf.keras.datasets.mnist.load_data()
x = x[:1000].astype('float32')[..., np.newaxis]
data = tf.data.Dataset.from_tensor_slices(x)

lae = models.LAE()
lae.compile(lv_learning_rate=5e-11, n_particles=5,
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
            preprocessor=tf.keras.layers.Normalization(axis=(0, 1)),
            postprocessor=tf.keras.layers.Normalization(axis=(0, 1),
                                                        invert=True),
            run_eagerly=True)
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = lae.fit(data=data, epochs=10, batch_size=256, callbacks=[tb])
print(history.history)

i = 4
samples = lae.decode_posterior_samples(n_samples=2, index=i)
images = [x[i, ..., 0]] + [samples[i, ...].numpy()[..., 0] for i in range(samples.shape[0])]
grid_size = math.ceil(len(images) ** (1/2))
for i, image in enumerate(images):
    ax = plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(255 - image.astype("uint8"), cmap='Greys')
    plt.axis("off")
plt.show()

a=1
#
# # Add particle and data indices to dataset:
# data = lae._add_indices_to_dataset(x, 2)
#
#
# # Shuffle and batch data:
# data = data.shuffle(1024).batch(4)
#
# for elem in data.take(1):
#     p_idx, d_idx, d_point = elem
#
# _train_latent_variables = tf.Variable(
#     initial_value=lae._prior.sample((2, 3)),
#     trainable=True)
# # latent_var_batch = tf.gather(self._train_latent_variables,
# #                                      indices=indices, axis=1)
#
# a=1