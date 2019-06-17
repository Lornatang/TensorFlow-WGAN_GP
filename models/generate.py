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
  model.add(layers.Dense(7 * 7 * 64, input_shape=(100, )))
  model.add(layers.Reshape((7, 7, 64)))
  model.add(layers.Conv2DTranspose(64, (3, 3),
                                   strides=(2, 2),
                                   padding='same',
                                   use_bias=False))
  model.add(layers.ReLU())
  model.add(layers.Conv2DTranspose(32, (3, 3),
                                   strides=(2, 2),
                                   padding='same',
                                   use_bias=False))
  model.add(layers.ReLU())
  model.add(layers.Conv2DTranspose(1, (3, 3),
                                   strides=(1, 1),
                                   padding='same',
                                   activation=tf.nn.sigmoid,
                                   use_bias=False))

  return model
