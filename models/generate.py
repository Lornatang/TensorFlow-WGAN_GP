# Copyright 2019 ChangyuLiu Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""implements discriminator network"""

import tensorflow as tf
from tensorflow.python.keras import layers


def make_generator_model():
  """ Generating network structure.

  Returns:
    Sequential model.

  """
  model = tf.keras.Sequential()
  model.add(layers.Dense(7 * 7 * 256,
                         use_bias=False,
                         input_shape=(100,)))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Reshape((7, 7, 256)))
  assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

  model.add(layers.Conv2DTranspose(128, (5, 5),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False))
  assert model.output_shape == (None, 7, 7, 128)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(64, (5, 5),
                                   strides=(2, 2),
                                   padding='same',
                                   use_bias=False))
  assert model.output_shape == (None, 14, 14, 64)
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU())

  model.add(layers.Conv2DTranspose(1, (5, 5),
                                   strides=(2, 2),
                                   padding='same',
                                   activation='tanh',
                                   use_bias=False))
  assert model.output_shape == (None, 28, 28, 1)

  return model
