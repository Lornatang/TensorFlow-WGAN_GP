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

"""Load fashion mnist datasets from here"""

import tensorflow as tf


def load_data():
  """

  Returns:
    tf.keras.datasets.fashion_mnist

  """
  TRAIN_BUF = 60000
  BATCH_SIZE = 16

  # load dataset
  (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

  # split dataset
  train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
    "float32"
  ) / 255.0

  # batch datasets
  train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
      .shuffle(TRAIN_BUF)
      .batch(BATCH_SIZE)
  )
  return train_dataset
