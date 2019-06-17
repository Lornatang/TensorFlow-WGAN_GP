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

import tensorflow as tf
import time
import os
import argparse

from data.datasets import load_data
from models.generate import make_generator_model
from models.discriminate import make_discriminator_model
from util.ops import compute_gradients, apply_gradients
from util.saver import save_checkpoints, generate_and_save_images

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=50, type=int,
                    help='Epochs for training.')
args = parser.parse_args()
print(args)

# define model save path
save_path = './training_checkpoint'

# create dir
if not os.path.exists(save_path):
  os.makedirs(save_path)

# define random noise
noise = tf.random.normal([16, 256])

# load dataset
train_dataset = load_data()

# load model network
generator = make_generator_model()
discriminator = make_discriminator_model()

# load model optim
gen_optim = tf.optimizers.Adam(0.0001, beta_1=0.5)
disc_optim = tf.optimizers.RMSprop(0.0005)  # train the model

checkpoint_dir, checkpoint, checkpoint_prefix = save_checkpoints(generator,
                                                                 discriminator,
                                                                 gen_optim,
                                                                 disc_optim,
                                                                 save_path)


# This annotation causes the function to be "compiled".
@tf.function
def train_step(img):
  gen_grad, disc_grad = compute_gradients(img)
  apply_gradients(gen_grad, disc_grad)


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

    print(f'Time for epoch {epoch + 1} is {time.time() - start:.3f} sec.')

  # Generate after the final epoch
  generate_and_save_images(generator,
                           epochs,
                           noise,
                           save_path)


if __name__ == '__main__':
  train(train_dataset, args.epochs)
