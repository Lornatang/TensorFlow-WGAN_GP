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

"""Implements G model and D model."""

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Reshape, Dense, Flatten, Dropout
from tensorflow.python.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.models import Model, Sequential


def generator(latent_dim=128):
  """ Generating network structure.

  Args:
    latent_dim: hidden neural unit.

  Returns:
    Sequential model.

  """
  model = Sequential()

  model.add(Dense(128 * 8 * 8,
                  activation=tf.nn.relu,
                  input_dim=latent_dim,
                  name='d1'))
  model.add(Reshape((8, 8, 128, ),
                    name='reshape1'))
  model.add(UpSampling2D(name='up1'))
  model.add(Conv2D(128,
                   kernel_size=4,
                   padding="same",
                   name='conv1'))
  model.add(BatchNormalization(momentum=0.8,
                               name='bn1'))
  model.add(Activation(tf.nn.relu,
                       name='act1'))
  model.add(UpSampling2D(name='up2'))
  model.add(Conv2D(64,
                   kernel_size=4,
                   padding="same",
                   name='conv2'))
  model.add(BatchNormalization(momentum=0.8,
                               name='bn2'))
  model.add(Activation(tf.nn.relu,
                       name='act2'))
  model.add(Conv2D(3,
                   kernel_size=4,
                   padding="same",
                   name='conv3'))
  model.add(Activation(tf.nn.tanh,
                       name='act3'))

  noise = Input(shape=(latent_dim, ))
  img = model(noise)

  model = Model(noise, img, name='Generator model')
  return model


def discriminator(img_shape=(32, 32, 3)):
  """ Discriminator network structure.

  Args:
    img_shape: img_height * img_width * channels.

  Returns:
    Sequential model.

  """
  model = Sequential()

  model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
  model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
  model.add(BatchNormalization(momentum=0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
  model.add(BatchNormalization(momentum=0.8))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(1))

  img = Input(shape=img_shape)
  validity = model(img)

  model = Model(img, validity, name='Discriminator model')
  return model
