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

"""The training loop begins with generator receiving a random seed as input."""

import os
import time

import tensorflow as tf

from data.datasets import load_data
from models.discriminate import make_discriminator_model
from models.generate import make_generator_model
from util.ops import gradient_penalty
from util.saver import generate_and_save_images, save_checkpoints

# define model save path
save_path = './training_checkpoint'

BUFFER_SIZE = 60000
BATCH_SIZE = 256

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, 1, 1, noise_dim])

# create dir
if not os.path.exists(save_path):
  os.makedirs(save_path)

# load dataset
train_dataset = load_data(BUFFER_SIZE, BATCH_SIZE)

# load model network
generator = make_generator_model()
discriminator = make_discriminator_model()

# load model optim
generator_optimizer = tf.optimizers.Adam(0.0001, beta_1=0.5)
discriminator_optimizer = tf.optimizers.RMSprop(0.0005)  # train the model

checkpoint_dir, checkpoint, checkpoint_prefix = save_checkpoints(generator,
                                                                 discriminator,
                                                                 generator_optimizer,
                                                                 discriminator_optimizer,
                                                                 save_path)


# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
  noise = tf.random.normal([BATCH_SIZE, 1, 1, noise_dim])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    # generator loss
    gen_loss = tf.reduce_mean(fake_output)

    # gradient penalty
    regularizer = gradient_penalty(discriminator, images, generated_images)

    # discriminator loss
    disc_loss = (tf.reduce_mean(real_output) -
                 tf.reduce_mean(fake_output) +
                 regularizer * 10.0)

  gradients_of_generator = gen_tape.gradient(gen_loss,
                                             generator.trainable_variables)
  gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                  discriminator.trainable_variables)

  generator_optimizer.apply_gradients(
    zip(gradients_of_generator, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(
    zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  """ train op
  Args:
    dataset: mnist dataset or cifar10 dataset.
    epochs: number of iterative training.
  """
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    generate_and_save_images(generator,
                             epoch + 1,
                             seed,
                             save_path)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print(f'Time for epoch {epoch + 1} is {time.time() - start:.3f} sec.')

  # Generate after the final epoch
  generate_and_save_images(generator,
                           epochs,
                           seed,
                           save_path)


if __name__ == '__main__':
  train(train_dataset, EPOCHS)
