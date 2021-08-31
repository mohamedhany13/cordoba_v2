import numpy as np

import general_methods
import lstm_att_layers
import tensorflow as tf
import os
import time
import logging
import graphviz
import pydot
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# type in NN architecture under investigation (this is to save different checkpoints for each architecture)
NN_arch = "autoregressive_attention"
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

normalize_dataset = True
input_length = 30
output_length = 1
input_features = 8
output_features = 1
validation_split = 0.2
test_split = 0.1
batch_size = 128
EPOCHS = 1000

lstm_units = 256
attention_units = 128
dropout_rate = 0.1

region = "Cordoba"
area_code = "06"
series = general_methods.load_dataset(region, area_code, normalize= normalize_dataset, swap= False, OS = OS,
                                      linux_path = linux_path)

x_train, y_train, x_dev, y_dev,\
x_test, y_test = general_methods.generate_train_dev_test_sets(series, validation_split, test_split,
                                                              input_length, output_length, area_code)
# Initialize optimizer and loss functions
if (learning_rate == 0):
    optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name= "train_loss")
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
dev_accuracy = tf.keras.metrics.Mean(name='dev_accuracy')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')
loss_object = tf.keras.losses.MeanSquaredError(reduction='none')

#h_dec_input = tf.Variable(tf.zeros((batch_size, output_length, lstm_units), dtype=tf.float32))
autoreg_att_model = lstm_att_layers.autoreg_att_model(output_features, lstm_units, attention_units, dropout_rate)
enc_dec_model = lstm_att_layers.enc_dec_model(output_features, lstm_units, attention_units, dropout_rate)
attention_model = lstm_att_layers.attention_model(output_features, lstm_units, attention_units, dropout_rate)
#dot_img_file = "C:\\Users\\moham\\Desktop\\model.png"
#tf.keras.utils.plot_model(enc_dec_model, to_file=dot_img_file, show_shapes=True)


checkpoint, checkpoint_prefix, ckpt_manager = general_methods.create_checkpoint(OS, NN_arch,
                                                                                autoreg_att_model, optimizer, linux_path)
# testing of model before training:
general_methods.test_model_w_heatmap(NN_arch, x_test, y_test, input_length, output_length, autoreg_att_model,
                                     test_accuracy, batch_size, lstm_units)

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
    # convert all inputs to train_step to tensors for faster processing
    #input_length_tensor = tf.convert_to_tensor(input_length)
    #output_length_tensor = tf.convert_to_tensor(output_length)
    # Training loop
    for epoch in range(EPOCHS):
        train_start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch in range(num_batches_train):
            batch_input, batch_output = general_methods.get_batch_data(batch, batch_size, x_train, y_train)

            if (NN_arch == "autoregressive_attention"):
                h_dec_input = tf.Variable(tf.zeros((batch_size, output_length, lstm_units), dtype=tf.float32))
                predicted_output, att_weights, \
                model_variables = lstm_att_layers.train_step(batch_input, batch_output, input_length, output_length,
                                                             autoreg_att_model, optimizer, train_loss, train_accuracy,
                                                             loss_object, h_dec_input)
            else:
                predicted_output, att_weights, \
                model_variables = lstm_att_layers.train_step(batch_input, batch_output, input_length, output_length,
                                                             autoreg_att_model, optimizer, train_loss, train_accuracy,
                                                             loss_object)
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
            if (NN_arch == "autoregressive_attention"):
                h_dec_input = tf.Variable(tf.zeros((batch_size, output_length, lstm_units), dtype=tf.float32))
                predicted_output, attention_weights = lstm_att_layers.evaluate(batch_input, batch_output,
                                                                               input_length, output_length,
                                                                               autoreg_att_model, dev_accuracy,
                                                                               h_dec_input)
            else:
                predicted_output, attention_weights = lstm_att_layers.evaluate(batch_input, batch_output,
                                                                               input_length, output_length,
                                                                               autoreg_att_model, dev_accuracy)

        dev_time = time.time() - dev_start
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}, \n'
              f'dev accuracy {dev_accuracy.result():.4f}')

        print(f"for 1 epoch: train time: {train_time:.2f} secs, dev time: {dev_time:.2f} secs")
        epoch_time = train_time + dev_time
        print(f'Time taken for 1 epoch: {epoch_time:.2f} secs, in minutes: {epoch_time/60:.2f} mins\n')

    total_train_time = time.time() - train_start_time
    # save total train time
    print("total training time = {} sec, in minutes: {} mins, in hours: {} hours".format(total_train_time, total_train_time/60, total_train_time/3600))

    # testing of model after training:
    general_methods.test_model_w_heatmap(NN_arch, x_test, y_test, input_length, output_length, autoreg_att_model,
                                         test_accuracy, batch_size, lstm_units)

 else:
    # testing
    # Restore the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))


x = 0