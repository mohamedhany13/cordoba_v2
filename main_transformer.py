import numpy as np

import general_methods
import transformer_layers
import tensorflow as tf
from general_methods import load_dataset, get_resnet_units, split_sequence_autoenc, \
    avg_batch_MAE, conv_tensor_array, split_dataframe_train_dev_test, split_sequence, \
    get_input_length, plot_pred_vs_target
import os
import time
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# type in NN architecture under investigation (this is to save different checkpoints for each architecture)
NN_arch = "transformer"
# choose OS type (linux or windows)
OS = "windows"
# choose 1 to use path in drive, 0 to use path in downloads (for linux only)
linux_path = 0
# choose whether to train the model or test it
# train ==> "train"
# test ==> "test"
# continue training with previous weights ==> "train_cont"
sim_mode = "train_cont"

normalize_dataset = True
input_length = 30
output_length = 7
input_features = 8
output_features = 1
validation_split = 0.2
test_split = 0.1
batch_size = 16
EPOCHS = 500

num_layers = 2
num_heads = 4
d_model = 64
# dff is number of units output from non-linear dense layer in the feed-forward block
dff = 32
dropout_rate = 0.1

region = "Cordoba"
area_code = "06"
series = general_methods.load_dataset(region, area_code, normalize= normalize_dataset, swap= False, OS = OS,
                                      linux_path = linux_path)

x_train, y_train, x_dev, y_dev,\
x_test, y_test = general_methods.generate_train_dev_test_sets(series, validation_split, test_split,
                                                              input_length, output_length, area_code)
# Initialize optimizer and loss functions
learning_rate = transformer_layers.CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
#optimizer = tf.keras.optimizers.Adam( learning_rate= 0.02)
train_loss = tf.keras.metrics.Mean(name= "train_loss")
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
dev_accuracy = tf.keras.metrics.Mean(name='dev_accuracy')
loss_object = tf.keras.losses.MeanSquaredError(reduction='none')

transformer = transformer_layers.Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_features=input_features,
    output_features=output_features,
    input_length=input_length,
    output_length=output_length,
    rate=dropout_rate)

checkpoint, checkpoint_prefix, ckpt_manager = general_methods.create_checkpoint(OS, NN_arch,
                                                                                transformer, optimizer, linux_path)
with tf.device('/gpu:0'):
 if (sim_mode == "train" or sim_mode == "train_cont"):

    if (sim_mode == "train_cont"):
        # Restore the latest checkpoint in checkpoint_dir
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))
        print("latest checkpoint loaded")

    # get number of batches for train set
    num_batches_train = general_methods.get_num_batches(x_train.shape[0], batch_size)
    # get number of batches of dev set
    num_batches_dev = general_methods.get_num_batches(x_dev.shape[0], batch_size)

    train_start_time = time.time()
    # Training loop
    for epoch in range(EPOCHS):
        train_start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch in range(num_batches_train):
            batch_input, batch_output = general_methods.get_batch_data(batch, batch_size, x_train, y_train)
            model_variables = transformer_layers.train_step(batch_input, batch_output, transformer,
                                          optimizer, train_loss, train_accuracy, loss_object)
            # check if model is training
            if (batch == 0):
                old_model_variables = model_variables
            else:
                any_change, variables_changed = general_methods.compare_lists(old_model_variables, model_variables)
            """
            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} MAPE {train_accuracy.result():.4f}')
            """

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        train_time = time.time() - train_start

        # evaluation
        dev_start = time.time()
        dev_accuracy.reset_states()
        for batch in range(num_batches_dev):
            batch_input, batch_output = general_methods.get_batch_data(batch, batch_size, x_dev, y_dev)
            dec_input = batch_output[:, -1:, :]
            evaluated_output = transformer_layers.evaluate(batch_input, batch_output, dec_input, transformer,
                                                           dev_accuracy)

        dev_time = time.time() - dev_start
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}, \n'
              f'dev accuracy {dev_accuracy.result():.4f}')

        print(f"for 1 epoch: train time: {train_time:.2f} secs, dev time: {dev_time:.2f} secs")
        epoch_time = train_time + dev_time
        print(f'Time taken for 1 epoch: {epoch_time:.2f} secs, in minutes: {epoch_time / 60:.2f} mins\n')

    total_train_time = time.time() - train_start_time
    # save total train time
    print("total training time = {} sec, in minutes: {} mins, in hours: {} hours".format(total_train_time,
                                                                                         total_train_time / 60,
                                                                                         total_train_time / 3600))

 else:
    # Restore the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))


x = 0