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

"""Load cifar10 datasets"""

import tensorflow as tf


def load_data():
  """ load every cats_vs_dogs dataset.

  Returns:
    dataset,

  """
  (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()

  train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
  train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

  train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
  train_dataset = train_dataset.shuffle(50000).batch(64)

  return train_dataset
