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
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# type in NN architecture under investigation (this is to save different checkpoints for each architecture)
NN_arch = "lstm"
# choose OS type (linux or windows)
OS = "linux"
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
input_length = 60
output_length = 1
input_features = 8
output_features = 1
validation_split = 0.2
test_split = 0.1
batch_size = 128
EPOCHS = 500

lstm_units = 256
attention_units = 128
dropout_rate = 0.1

region = "Cordoba"
area_code = "06"
series = general_methods.load_dataset(region, area_code, normalize= normalize_dataset, swap= False, OS = OS,
                                      linux_path = linux_path)

normal_series = series["Co" + area_code + "ETo"]
normal_series_block_shuffled = general_methods.block_shuffle_series(normal_series, block_shuffle_size)
normal_series_block_shuffled = np.squeeze(normal_series_block_shuffled)
random_series = normal_series.sample(frac=1, random_state=1).reset_index(drop=True)
random_series_block_shuffled = general_methods.block_shuffle_series(random_series, block_shuffle_size)
random_series_block_shuffled = np.squeeze(random_series_block_shuffled)

normal_series_list = normal_series.to_list()
normal_bs_series_list = normal_series_block_shuffled.to_list()
random_series_list = random_series.to_list()
random_bs_series_list = random_series_block_shuffled.to_list()

n_x_train, n_y_train, n_x_dev, n_y_dev, _, _ = general_methods.generate_datasets_univariate(normal_series_list, validation_split,
                                                                                            test_split, input_length,
                                                                                            output_length)
n_b_x_train, n_b_y_train, n_b_x_dev, n_b_y_dev, _, _ = general_methods.generate_datasets_univariate(normal_bs_series_list, validation_split,
                                                                                            test_split, input_length,
                                                                                            output_length)
r_x_train, r_y_train, r_x_dev, r_y_dev, _, _ = general_methods.generate_datasets_univariate(random_series_list, validation_split,
                                                                                            test_split, input_length,
                                                                                            output_length)
r_b_x_train, r_b_y_train, r_b_x_dev, r_b_y_dev, _, _ = general_methods.generate_datasets_univariate(random_bs_series_list, validation_split,
                                                                                            test_split, input_length,
                                                                                            output_length)
model = lstm_att_layers.lstm_model(lstm_units, output_features)

optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name= "train_loss")
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
dev_accuracy = tf.keras.metrics.Mean(name='dev_accuracy')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')
loss_object = tf.keras.losses.MeanSquaredError(reduction='none')

checkpoint, checkpoint_prefix, ckpt_manager = general_methods.create_checkpoint(OS, NN_arch,
                                                                                model, optimizer, linux_path)
all_SSE = []
with tf.device('/gpu:0'):
    x_train, y_train, x_dev, y_dev = n_x_train, n_y_train, n_x_dev, n_y_dev
    lstm_att_layers.training_function_lstm(sim_mode, checkpoint, checkpoint_prefix, ckpt_manager, batch_size, EPOCHS,
                                           train_loss, optimizer, loss_object, train_accuracy, dev_accuracy, model,
                                           input_length, output_length, x_train, y_train, x_dev, y_dev)
    batch_input, batch_output = general_methods.get_batch_data(0, batch_size, x_dev, y_dev)
    y_pred = lstm_att_layers.evaluate_lstm(batch_input, batch_output, input_length, output_length, model, dev_accuracy)
    normal_SSE = general_methods.calc_SSE(batch_output, y_pred)
    all_SSE.append(normal_SSE)

    x_train, y_train, x_dev, y_dev = n_b_x_train, n_b_y_train, n_b_x_dev, n_b_y_dev
    lstm_att_layers.training_function_lstm(sim_mode, checkpoint, checkpoint_prefix, ckpt_manager, batch_size, EPOCHS,
                                           train_loss, optimizer, loss_object, train_accuracy, dev_accuracy, model,
                                           input_length, output_length, x_train, y_train, x_dev, y_dev)
    batch_input, batch_output = general_methods.get_batch_data(0, batch_size, x_dev, y_dev)
    y_pred = lstm_att_layers.evaluate_lstm(batch_input, batch_output, input_length, output_length, model, dev_accuracy)
    normal_bs_SSE = general_methods.calc_SSE(batch_output, y_pred)
    all_SSE.append(normal_bs_SSE)

    x_train, y_train, x_dev, y_dev = r_x_train, r_y_train, r_x_dev, r_y_dev
    lstm_att_layers.training_function_lstm(sim_mode, checkpoint, checkpoint_prefix, ckpt_manager, batch_size, EPOCHS,
                                           train_loss, optimizer, loss_object, train_accuracy, dev_accuracy, model,
                                           input_length, output_length, x_train, y_train, x_dev, y_dev)
    batch_input, batch_output = general_methods.get_batch_data(0, batch_size, x_dev, y_dev)
    y_pred = lstm_att_layers.evaluate_lstm(batch_input, batch_output, input_length, output_length, model, dev_accuracy)
    random_SSE = general_methods.calc_SSE(batch_output, y_pred)
    all_SSE.append(random_SSE)

    x_train, y_train, x_dev, y_dev = r_b_x_train, r_b_y_train, r_b_x_dev, r_b_y_dev
    lstm_att_layers.training_function_lstm(sim_mode, checkpoint, checkpoint_prefix, ckpt_manager, batch_size, EPOCHS,
                                           train_loss, optimizer, loss_object, train_accuracy, dev_accuracy, model,
                                           input_length, output_length, x_train, y_train, x_dev, y_dev)
    batch_input, batch_output = general_methods.get_batch_data(0, batch_size, x_dev, y_dev)
    y_pred = lstm_att_layers.evaluate_lstm(batch_input, batch_output, input_length, output_length, model, dev_accuracy)
    random_bs_SSE = general_methods.calc_SSE(batch_output, y_pred)
    all_SSE.append(random_bs_SSE)

    if (OS == "linux"):
        if (linux_path == 1):
            file_path = r"/media/hamamgpu/Drive3/mohamed-hany/SSE.txt"
        else:
            file_path = r"/home/mohamed-hany/Downloads/SSE.txt"
    else:
        file_path = r"C:\Users\moham\Desktop\masters\master_thesis\time_series_analysis\model_testing\SSE.txt"

    f = open(file_path, "w")
    all_SSE = str(all_SSE)
    f.write(all_SSE)
    f.close()

x= 0
