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

"""Some operations to implement gradients"""

import tensorflow as tf


def gradient_penalty(discriminator, real_img, fake_img):
  """ Only a gradient penalty is applied to the discriminator.

  Args:
    discriminator: discriminator model.
    real_img: Real input image data.
    fake_img: Fake input image data.

  Returns:
    discriminate regularizer.

  """
  epsilon = tf.random.uniform([real_img.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)
  x_hat = epsilon * real_img + (1 - epsilon) * fake_img

  with tf.GradientTape() as gp:
    gp.watch(x_hat)
    d_hat = discriminator(x_hat)

  gradients = gp.gradient(d_hat, x_hat)
  ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
  regularizer = tf.reduce_mean((ddx - 1.0) ** 2)

  return regularizer
