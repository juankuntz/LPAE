import models
import tensorflow as tf
import numpy as np


def test_lae_add_indices_to_dataset():
    lae = models.LAE()
    data = tf.data.Dataset.from_tensor_slices(np.array(['A', 'B', 'C']))
    data_with_idx = [(0, 0, b'A'), (0, 1, b'B'), (0, 2, b'C'),
                     (1, 0, b'A'), (1, 1, b'B'), (1, 2, b'C')]
    assert (list(lae._add_indices_to_dataset(data, 2).as_numpy_iterator())
            == data_with_idx)
