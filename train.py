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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from tqdm.autonotebook import tqdm

TRAIN_BUF = 60000
BATCH_SIZE = 512
TEST_BUF = 10000
DIMS = (28, 28, 1)
N_TRAIN_BATCHES = int(TRAIN_BUF / BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUF / BATCH_SIZE)

# load dataset
(train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()

# split dataset
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
  "float32"
) / 255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32") / 255.0

# batch datasets
train_dataset = (
  tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(TRAIN_BUF)
    .batch(BATCH_SIZE)
)
test_dataset = (
  tf.data.Dataset.from_tensor_slices(test_images)
    .shuffle(TEST_BUF)
    .batch(BATCH_SIZE)
)


class WGAN(tf.keras.Model):
  """[summary]
  I used github/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/ as a reference on this.

  Extends:
      tf.keras.Model
  """

  def __init__(self, **kwargs):
    super(WGAN, self).__init__()
    self.__dict__.update(kwargs)

    self.gen = tf.keras.Sequential(self.gen)
    self.disc = tf.keras.Sequential(self.disc)

  def generate(self, z):
    return self.gen(z)

  def discriminate(self, x):
    return self.disc(x)

  def compute_loss(self, x):
    """ passes through the network and computes loss
    """
    # pass through network
    # generating noise from a uniform distribution

    z_samp = tf.random.normal([x.shape[0], 1, 1, self.n_Z])

    # run noise through generator
    x_gen = self.generate(z_samp)
    # discriminate x and x_gen
    logits_x = self.discriminate(x)
    logits_x_gen = self.discriminate(x_gen)

    # gradient penalty
    d_regularizer = self.gradient_penalty(x, x_gen)
    # losses
    disc_loss = (
            tf.reduce_mean(logits_x)
            - tf.reduce_mean(logits_x_gen)
            + d_regularizer * self.gradient_penalty_weight
    )

    # losses of fake with label "1"
    gen_loss = tf.reduce_mean(logits_x_gen)

    return disc_loss, gen_loss

  def compute_gradients(self, x):
    """ passes through the network and computes loss
    """
    # pass through network
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      disc_loss, gen_loss = self.compute_loss(x)

    # compute gradients
    gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)

    disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

    return gen_gradients, disc_gradients

  def apply_gradients(self, gen_gradients, disc_gradients):
    self.gen_optimizer.apply_gradients(
      zip(gen_gradients, self.gen.trainable_variables)
    )
    self.disc_optimizer.apply_gradients(
      zip(disc_gradients, self.disc.trainable_variables)
    )

  def gradient_penalty(self, x, x_gen):
    epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
    x_hat = epsilon * x + (1 - epsilon) * x_gen
    with tf.GradientTape() as t:
      t.watch(x_hat)
      d_hat = self.discriminate(x_hat)
    gradients = t.gradient(d_hat, x_hat)
    ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
    d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
    return d_regularizer

  @tf.function
  def train(self, train_x):
    gen_gradients, disc_gradients = self.compute_gradients(train_x)
    self.apply_gradients(gen_gradients, disc_gradients)


N_Z = 64

generator = [
  tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
  tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
  tf.keras.layers.Conv2DTranspose(
    filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
  ),
  tf.keras.layers.Conv2DTranspose(
    filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
  ),
  tf.keras.layers.Conv2DTranspose(
    filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
  ),
]

discriminator = [
  tf.keras.layers.InputLayer(input_shape=DIMS),
  tf.keras.layers.Conv2D(
    filters=32, kernel_size=3, strides=(2, 2), activation="relu"
  ),
  tf.keras.layers.Conv2D(
    filters=64, kernel_size=3, strides=(2, 2), activation="relu"
  ),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units=1, activation="sigmoid"),
]

# optimizers
gen_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.RMSprop(0.0005)  # train the model
# model
model = WGAN(
  gen=generator,
  disc=discriminator,
  gen_optimizer=gen_optimizer,
  disc_optimizer=disc_optimizer,
  n_Z=N_Z,
  gradient_penalty_weight=10.0
)


# exampled data for plotting results
def plot_reconstruction(models, nex=8, zm=2):
  samples = models.generate(tf.random.normal(shape=(BATCH_SIZE, N_Z)))
  fig, axs = plt.subplots(ncols=nex, nrows=1, figsize=(zm * nex, zm))
  for axi in range(nex):
    axs[axi].matshow(
      samples.numpy()[axi].squeeze(), cmap='Greys', vmin=0, vmax=1
    )
    axs[axi].axis('off')
  plt.show()


# a pandas dataframe to save the loss information to
losses = pd.DataFrame(columns=['disc_loss', 'gen_loss'])

n_epochs = 200
for epoch in range(n_epochs):
  # train
  for batch, train_x in tqdm(
          zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
  ):
    model.train(train_x)
  # test on holdout
  loss = []
  for batch, test_x in tqdm(
          zip(range(N_TEST_BATCHES), test_dataset), total=N_TEST_BATCHES
  ):
    loss.append(model.compute_loss(train_x))
  losses.loc[len(losses)] = np.mean(loss, axis=0)
  # plot results
  display.clear_output()
  print(
    "Epoch: {} | disc_loss: {} | gen_loss: {}".format(
      epoch, losses.disc_loss.values[-1], losses.gen_loss.values[-1]
    )
  )
  plot_reconstruction(model)

plt.plot(losses.gen_loss.values)

plt.plot(losses.disc_loss.values)
