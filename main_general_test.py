import custom_layers_models
import tensorflow as tf
from general_methods import load_dataset, get_resnet_units, split_sequence_autoenc, \
    avg_batch_MAE, conv_tensor_array, split_dataframe_train_dev_test, split_sequence, \
    get_input_length
from evaluate_methods import evaluate_CRNN_auto_enc, evaluate_CRNN
from fit_methods import train_step_CRNN_auto_enc, train_step_CRNN
import os
import time

input_window = 10
input_years = 1
input_length, months_per_year, num_windows_per_month = get_input_length(input_window, input_years)
output_length = 3
input_features = 8
output_features = 1
validation_split = 0.3
test_split = 0
batch_size = 64
EPOCHS = 700
kernel_size= 4
num_kernels= 64
pool_size= 2
pool_strides= 1
rnn_units = 128
att_dense_units = 12
att_dense_units_1st = 128
att_dense_units_2nd = 64
att_kernel_size = 3
att_num_kernels = 32
att_mon_year_dense_units = 64
monthly_kernel_size, monthly_num_kernels, monthly_pool_size = 2, 64, 1
yearly_kernel_size, yearly_num_kernels, yearly_pool_size = 2, 64, 1
resnet_ffinput_units= get_resnet_units(input_features,input_window, kernel_size, num_kernels, pool_size)

region = "Cordoba"
area_code = "06"
series = load_dataset(region, area_code, normalize= True)

# split data into training set, development set, test set
train_set, dev_set, test_set = split_dataframe_train_dev_test(series, validation_split, test_split)

# create the training set
x_train, x_target_train, y_train = split_sequence(train_set, input_length, output_length, area_code)
# create the dev set
x_dev, x_target_dev, y_dev = split_sequence(dev_set, input_length, output_length, area_code)
# create the test set
x_test, x_target_test, y_test = split_sequence(test_set, input_length, output_length, area_code)

# Initialize optimizer and loss functions
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.MeanSquaredError(reduction='none')

encoder = custom_layers_models.CRNN_encoder(rnn_units, kernel_size, num_kernels,
                                            pool_size, resnet_ffinput_units)
decoder = custom_layers_models.CRNN_decoder(rnn_units, output_features,
                                            input_window, input_years, months_per_year, num_windows_per_month,
                                            att_dense_units, att_dense_units_1st, att_dense_units_2nd,
                                            att_mon_year_dense_units,
                                            monthly_kernel_size, monthly_num_kernels, monthly_pool_size,
                                            yearly_kernel_size, yearly_num_kernels, yearly_pool_size)

# create a checkpoint instance
checkpoint_dir = r"C:\Users\moham\Desktop\masters\master_thesis\time_series_analysis\model_testing\cordoba checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt.txt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

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

    # Initialize the hidden state
    # enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    total_MAE = 0
    total_MAPE = 0

    # Loop through the dataset
    for batch in range(num_batches_train):

        # get the current batch of x_train & y_train
        start_index = batch * batch_size
        end_index = start_index + batch_size
        if (end_index > x_train.shape[0]):
            end_index = x_train.shape[0]
        batch_input = x_train[start_index: end_index]
        batch_target_input = x_target_train[start_index: end_index]
        batch_output = y_train[start_index: end_index]
        #batch_output = tf.expand_dims(batch_output, axis=1)

        # Call the train method
        output_pred, batch_loss, trainable_variables = train_step_CRNN(batch_input, batch_target_input, batch_output,
                                                                       encoder, decoder, optimizer,
                                                                       loss_object, rnn_units, input_window,
                                                                       input_length, output_features)

        output_pred_array = conv_tensor_array(output_pred)
        # compute MAE & MAPE (per epoch)
        MAE , MAPE = avg_batch_MAE(batch_output, output_pred_array)
        total_MAE += MAE
        total_MAPE += MAPE
        # Compute the loss (per epoch)
        total_loss += batch_loss

        """
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))
        """

    # Save (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    dev_total_MAE = 0
    dev_total_MAPE = 0
    for batch in range(num_batches_dev):

        # get the current batch of x_train & y_train
        start_index = batch * batch_size
        end_index = start_index + batch_size
        if (end_index > x_dev.shape[0]):
            end_index = x_dev.shape[0]
        batch_input = x_dev[start_index: end_index]
        batch_target_input = x_target_dev[start_index: end_index]
        batch_output = y_dev[start_index: end_index]

        output_dev, batch_MAE_dev, batch_MAPE_dev = evaluate_CRNN(batch_input, batch_target_input, batch_output,
                                                                           encoder, decoder,
                                                                           rnn_units, input_window,
                                                                           input_length, output_features)

        dev_total_MAE += batch_MAE_dev
        dev_total_MAPE += batch_MAPE_dev

    # Output the loss observed until that epoch
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / num_batches_train))
    print('Epoch {} train_MAE: {} , train_MAPE: {} & dev_MAE : {} , dev_MAPE: {}'.format(epoch + 1,
                                            total_MAE/num_batches_train, total_MAPE/num_batches_train,
                                            dev_total_MAE/num_batches_dev, dev_total_MAPE/num_batches_dev))

    print('Time taken for 1 epoch {} sec, {} min\n'.format((time.time() - start), (time.time() - start) / 60))

total_train_time = time.time() - train_start_time
print("total training time = {}".format(total_train_time))

x = 0