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

"""save model function"""

import tensorflow as tf

import os
from matplotlib import pyplot as plt


def save_checkpoints(generator, discriminator, generator_optimizer, discriminator_optimizer, save_path):
  """ save gan model

  Args:
    generator: generate model.
    discriminator: discriminate model.
    generator_optimizer: generate optimizer func.
    discriminator_optimizer: discriminator optimizer func.
    save_path: save gan model dir path.

  Returns:
    checkpoint path

  """
  checkpoint_dir = save_path
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator,
                                   discriminator=discriminator)

  return checkpoint_dir, checkpoint, checkpoint_prefix


def generate_and_save_images(model, epoch, seed, save_path):
  """ make sure the training parameter is set to False because we
     don't want to train the batch-norm layer when doing inference.

  Args:
    model: generate model.
    epoch: train epoch nums.
    seed: random seed at (64, 128).
    save_path: generate images path.

  Returns:
    none.

  """
  predictions = model(seed, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :] * 255.0 + 255.0)
    plt.axis('off')

  plt.savefig(save_path + '/' + f'{epoch:04d}.png')
  plt.close(fig)
