import numpy as np

import general_methods
import lstm_att_layers
import tensorflow as tf
import os
import time
import pandas as pd
import logging
import graphviz
import pydot
from tensorflow import keras
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# type in NN architecture under investigation (this is to save different checkpoints for each architecture)
NN_arch = "lstm"
# choose OS type (linux or windows)
OS = "windows"
# choose 1 to use path in drive, 0 to use path in downloads (for linux only)
linux_path = 0
# choose whether to train the model or test it
# train ==> "train"
# test ==> "test"
# continue training with previous weights ==> "train_cont"
sim_mode = "train"

# choose zero if you want to set it to default
learning_rate = 0.001
block_shuffle_size = 10

normalize_dataset = False
input_length = 30
output_length = 1
input_features = 8
output_features = 1
validation_split = 0.2
test_split = 0.1
batch_size = 64
EPOCHS = 500

lstm_units = 64
attention_units = 128
dropout_rate = 0.1

region = "Cordoba"
area_code = "06"
series = general_methods.load_dataset(region, area_code, normalize= normalize_dataset, swap= False, OS = OS,
                                      linux_path = linux_path)

normal_series = series["Co" + area_code + "ETo"]
normal_series_list = normal_series.to_list()
#series_array = np.array(normal_series_list)
#series_array = series_array[..., np.newaxis]

n_x_train, n_y_train, _, _, _, _ = general_methods.generate_datasets_univariate(normal_series_list, 0,
                                                                                            0, input_length,
                                                                                            output_length)
n_y_train = n_y_train[..., -1]
"""
input_data = series_array[:-input_length]
targets = series_array[input_length:]
dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    input_data, targets, sequence_length=input_length, batch_size= batch_size)

for batch in dataset.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)
"""

inputs = keras.layers.Input(shape=(n_x_train.shape[1], n_x_train.shape[2]))
lstm_out = keras.layers.LSTM(lstm_units)(inputs)
outputs = keras.layers.Dense(output_features)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse",
              metrics=["mean_absolute_percentage_error"])
model.summary()

model.fit(n_x_train, n_y_train, batch_size = batch_size, epochs=EPOCHS, validation_split= validation_split,
          verbose = 1)


x = 0