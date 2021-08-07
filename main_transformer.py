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
# choose whether to train the model or test it
# train ==> "train"
# test ==> "test"
# continue training with previous weights ==> "train_cont"
sim_mode = "train"

normalize_dataset = True
input_length = 365
output_length = 30
input_features = 8
output_features = 1
validation_split = 0.2
test_split = 0.1
batch_size = 64
EPOCHS = 500

num_layers = 3
num_heads = 4
d_model = 128
# dff is number of units output from non-linear dense layer in the feed-forward block
dff = 64
dropout_rate = 0.1

region = "Cordoba"
area_code = "06"
series = load_dataset(region, area_code, normalize= normalize_dataset, swap= False, OS = OS)
# randomize the training set to better train the model instead of having nearly similar training examples in each batch
series_shuffled = series.sample(frac = 1)
# split data into training set, development set, test set
train_set, dev_set, test_set = split_dataframe_train_dev_test(series_shuffled, validation_split, test_split)

# create the training set
x_train, x_target_train, y_train = split_sequence(train_set, input_length, output_length, area_code)
y_train = y_train[..., np.newaxis]
# create the dev set
x_dev, x_target_dev, y_dev = split_sequence(dev_set, input_length, output_length, area_code)
y_dev = y_dev[..., np.newaxis]
# create the test set
x_test, x_target_test, y_test = split_sequence(test_set, input_length, output_length, area_code)
y_test = y_test[..., np.newaxis]

# Initialize optimizer and loss functions
learning_rate = transformer_layers.CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
#optimizer = tf.keras.optimizers.Adam( learning_rate= 0.02)
train_loss = tf.keras.metrics.Mean(name= "train_loss")
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
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

# create a checkpoint instance
if (OS == "linux"):
    checkpoint_dir = r"/media/hamamgpu/Drive3/mohamed-hany/cordoba_ckpts"
else:
    checkpoint_dir = r"C:\Users\moham\Desktop\masters\master_thesis\time_series_analysis\model_testing\cordoba checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_" + NN_arch + ".txt")
checkpoint = tf.train.Checkpoint(transformer = transformer, optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=5)

with tf.device('/gpu:0'):
 if (sim_mode == "train" or sim_mode == "train_cont"):

    if (sim_mode == "train_cont"):
        # Restore the latest checkpoint in checkpoint_dir
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("latest checkpoint loaded")

    # get number of batches for train set
    num_batches_train = int(x_train.shape[0] / batch_size)
    remainder_train = x_train.shape[0] % batch_size
    if (remainder_train != 0):
        num_batches_train += 1

    # get number of batches of dev set
    num_batches_dev = int(x_dev.shape[0] / batch_size)
    remainder_dev = x_dev.shape[0] % batch_size
    if (remainder_dev != 0):
        num_batches_dev += 1

    train_start_time = time.time()
    # Training loop
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch in range(num_batches_train):
            batch_input, batch_output = general_methods.get_batch_data(batch, batch_size, x_train, y_train)
            model_variables = transformer_layers.train_step(batch_input, batch_output, transformer,
                                          optimizer, train_loss, train_accuracy, loss_object)

            if batch % 50 == 0:
                print(
                    f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} MAPE {train_accuracy.result():.4f}')

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs, in minutes: {(time.time() - start)/60:.2f} mins\n')

    total_train_time = time.time() - train_start_time
    print("total training time = {}".format(total_train_time))

 else:
    # Restore the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


x = 0