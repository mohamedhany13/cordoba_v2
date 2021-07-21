from general_methods import load_dataset, split_sequence, train_step, forecast, avg_batch_MAE,\
    split_sequence_autoenc, train_step_v1, evaluate_autoenc
from custom_layers_models import Encoder, Decoder, CRNN_encoder, CRNN_auto_decoder
import tensorflow as tf
import os
import time

region = "Cordoba"
area_code = "06"
CNN_filter_size = 4
CNN_num_filters = 128
RNN_num_units = 256
Dense_num_units = 128
Dense_activation = "LeakyReLU"
n_output_features = 1
enc_regularization = "l2"
enc_dropout = 0.2
#input_length = 1820 #(nearest number to 5*365 which is divisible by 7)
input_length = 21
output_length = input_length
batch_size = 64
EPOCHS = 50
validation_set_ratio = 0.3

# load the dataset
series = load_dataset(region = region, area_code = area_code, normalize= "True")

# Initialize optimizer and loss functions
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
#optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.MeanSquaredError(reduction='none')


# create encoder & decoder instances
encoder = CRNN_encoder(CNN_filter_size= CNN_filter_size, CNN_num_filters= CNN_num_filters,
                       RNN_num_units= RNN_num_units, Dense_num_units= Dense_num_units,
                       Dense_activation= Dense_activation, enc_regularization= enc_regularization,
                       enc_dropout= enc_dropout)
decoder = CRNN_auto_decoder(RNN_num_units= RNN_num_units, Dense_activation= Dense_activation,
                            n_output_features= n_output_features)
"""
encoder = tf.keras.layers.LSTM(RNN_num_units)
decoder = tf.keras.layers.LSTM()
"""

# create a checkpoint instance
checkpoint_dir = r"C:\Users\moham\Desktop\masters\master_thesis\time_series_analysis\model_testing\cordoba checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt.txt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# create the training set & validation set
training_set_range = int((1 - validation_set_ratio) * len(series))
training_set = series[0: training_set_range]
x_train, y_train = split_sequence_autoenc(training_set, input_length, area_code = area_code)
validation_set = series[training_set_range: len(series)]
x_validation, y_validation = split_sequence_autoenc(validation_set, input_length, area_code = area_code)

# get number of batches
num_batches = int(x_train.shape[0] / batch_size)
remainder = x_train.shape[0] % batch_size
if (remainder != 0):
    num_batches += 1

# get number of weeks
num_weeks = int(x_train.shape[1]/7)

#y_validation_pred, avg_MAE = evaluate_autoenc(x_validation, y_validation, encoder, decoder, num_weeks)
training_start_time = time.time()
# Training loop
for epoch in range(EPOCHS):
    start = time.time()

    # Initialize the hidden state
    # enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    # Loop through the dataset
    for batch in range(num_batches):

        # get the current batch of x_train & y_train
        start_index = batch * batch_size
        end_index = start_index + batch_size
        if (end_index > x_train.shape[0]):
            end_index = x_train.shape[0]
        batch_input = x_train[start_index: end_index]
        batch_output = y_train[start_index: end_index]
        #batch_output = tf.expand_dims(batch_output, axis=1)

        # Call the train method
        output_pred, batch_loss, trainable_variables = train_step_v1(batch_input, batch_output,
                                                                     encoder, decoder, optimizer,
                                                                     loss_object, num_weeks, RNN_num_units)

        # Compute the loss (per epoch)
        total_loss += batch_loss

#        if batch % 100 == 0:
#            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

    # Save (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    # test with validation set
    y_validation_pred, avg_MAE, avg_MAPE = evaluate_autoenc(x_validation, y_validation, encoder, decoder, num_weeks)
    # Output the loss observed until that epoch
    print('Epoch {} Loss {:.4f}, validation MAE: {}, MAPE: {}'.format(epoch + 1,
                                                                      total_loss / num_batches, avg_MAE, avg_MAPE))

    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# get whole training time
train_time = time.time() - training_start_time
print("training time : {} sec\n".format(train_time))
# Restore the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
x_test = tf.expand_dims(x_train[0], axis=0)
y_test = tf.expand_dims(y_train[0], axis=0)
#MAE = forecast(x_test, y_test, encoder, decoder)

x=0