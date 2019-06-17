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

"""train datasets for wgan.

The training loop begins with generator receiving a random seed as input.
   That seed is used to produce an image.
   The discriminator is then used to classify real images (drawn from the training set)
   and fakes images (produced by the generator).
   The loss is calculated for each of these models,
   and the gradients are used to update the generator and discriminator.
"""

import argparse
import os
import time

import tensorflow as tf

from old.data.datasets import load_data
from old.models.discriminate import make_discriminator_model
from old.models.generate import make_generator_model
from old.util.losses import discriminator_loss, generator_loss
from old.util.optim import discriminator_optimizer, generator_optimizer
from old.util.saver import generate_and_save_images, save_checkpoints

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=50, type=int,
                    help='Epochs for training.')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Every batch size.')
parser.add_argument('--LAMBDA', default=10, type=int,
                    help='none')
args = parser.parse_args()
print(args)

# define model save path
save_path = './training_checkpoint'

# create dir
if not os.path.exists(save_path):
  os.makedirs(save_path)

# define random noise
noise = tf.random.normal([64, 128])

# load dataset
train_dataset = load_data()

# load network and optim paras
generator = make_generator_model()
generator_optimizer = generator_optimizer()

discriminator = make_discriminator_model()
discriminator_optimizer = discriminator_optimizer()

checkpoint_dir, checkpoint, checkpoint_prefix = save_checkpoints(generator,
                                                                 discriminator,
                                                                 generator_optimizer,
                                                                 discriminator_optimizer,
                                                                 save_path)

# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
  """ break it down into training steps.
  Args:
    images: input images.
  """
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)

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
                             noise,
                             save_path)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print(f'Time for epoch {epoch+1} is {time.time()-start:.3f} sec.')

  # Generate after the final epoch
  generate_and_save_images(generator,
                           epochs,
                           noise,
                           save_path)


train(train_dataset, args.epochs)
