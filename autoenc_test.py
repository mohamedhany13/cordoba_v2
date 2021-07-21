import custom_layers_models
import tensorflow as tf
from general_methods import load_dataset, get_resnet_units, split_sequence_autoenc, \
    avg_batch_MAE, conv_tensor_array, split_dataframe_train_dev_test
from evaluate_methods import forecast_CRNN_auto_enc
from fit_methods import train_step_CRNN_auto_enc
import os
import time

validation_split = 0.3
test_split = 0
batch_size = 64
EPOCHS = 500
input_length = 21
output_length = input_length
input_window = 7
input_features = 8
output_features = 1
kernel_size= 4
num_kernels= 64
pool_size= 2
pool_strides= 1
rnn_units = 128
max_attention_span = 1
resnet_ffinput_units= get_resnet_units(input_features,input_window, kernel_size, num_kernels, pool_size)

region = "Cordoba"
area_code = "06"
series = load_dataset(region, area_code)

# split data into training set, development set, test set
train_set, dev_set, test_set = split_dataframe_train_dev_test(series, validation_split, test_split)
# create the training set
x_train, x_target_train, y_train = split_sequence_autoenc(train_set, input_length, area_code)
# create the dev set
x_dev, x_target_dev, y_dev = split_sequence_autoenc(dev_set, input_length, area_code)
# create the test set
x_test, x_target_test, y_test = split_sequence_autoenc(test_set, input_length, area_code)

optimizer = tf.keras.optimizers.Adam()
encoder = custom_layers_models.CRNN_encoder(rnn_units, kernel_size, num_kernels,
                                            pool_size, resnet_ffinput_units)
decoder = custom_layers_models.CRNN_auto_decoder(rnn_units, output_features)

# create a checkpoint instance
checkpoint_dir = r"C:\Users\moham\Desktop\masters\master_thesis\time_series_analysis\model_testing\cordoba checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt.txt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# Restore the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

output_dev, predicted_APE = forecast_CRNN_auto_enc(x_dev[0], x_target_dev[0],
                                                   encoder, decoder,
                                                   rnn_units, input_window,
                                                   input_length, max_attention_span)

x = 0

