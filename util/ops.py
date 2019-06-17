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

from models.generate import make_generator_model
from models.discriminate import make_discriminator_model

gradient_penalty_weight = 10.0
dim = 100

generator = make_generator_model()
discriminator = make_discriminator_model()

gen_optim = tf.optimizers.Adam(0.0001, beta_1=0.5)
disc_optim = tf.optimizers.RMSprop(0.0005)  # train the model


def compute_loss(real_img):
  """ passes through the network and computes loss
  """
  # pass through network
  # generating noise from a uniform distribution

  z = tf.random.normal([real_img.shape[0], 1, 1, dim])

  # run noise through generator
  generated_images = generator(z)

  # discriminate real img and fake img
  real_output = discriminator(real_img)
  fake_output = discriminator(generated_images)

  # gradient penalty
  regularizer = gradient_penalty(real_img, generated_images)
  # losses
  loss_of_discriminator = (tf.reduce_mean(real_output) -
                           tf.reduce_mean(fake_output) +
                           regularizer * gradient_penalty_weight)

  # losses of fake with label "1"
  loss_of_generator = tf.reduce_mean(fake_output)

  return loss_of_generator, loss_of_discriminator


def compute_gradients(img):
  """ passes through the network and computes loss
  """
  # pass through network
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_loss, disc_loss = compute_loss(img)

  # compute gradients
  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  return gradients_of_generator, gradients_of_discriminator


def gradient_penalty(real_img, fake_img):
  """ Only a gradient penalty is applied to the discriminator.

  Args:
    real_img: Real input image data.
    fake_img: Fake input image data.

  Returns:
    discriminate regularizer.

  """
  epsilon = tf.random.uniform([real_img.shape[0], 1, 1, 1], 0.0, 1.0)
  x_hat = epsilon * real_img + (1 - epsilon) * fake_img

  with tf.GradientTape() as gp:
    gp.watch(x_hat)
    d_hat = discriminator(x_hat)

  gradients = gp.gradient(d_hat, x_hat)
  ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
  regularizer = tf.reduce_mean((ddx - 1.0) ** 2)

  return regularizer


def apply_gradients(gen_grad, disc_grad):
  """ Achieve gradient training.

  Args:
    gen_grad: Generator gradient.
    disc_grad: Discriminator gradient.

  Returns:
    None.
  """
  gen_optim.apply_gradients(zip(gen_grad, generator.trainable_variables))
  disc_optim.apply_gradients(zip(disc_grad, discriminator.trainable_variables))
