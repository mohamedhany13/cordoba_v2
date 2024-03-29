import custom_layers_models
import tensorflow as tf
from general_methods import load_dataset, get_resnet_units, split_sequence_autoenc, \
    avg_batch_MAE, conv_tensor_array, split_dataframe_train_dev_test, split_sequence, \
    get_input_length, plot_pred_vs_target
from evaluate_methods import evaluate_CRNN_auto_enc, evaluate_CRNN, evaluate_attention
from fit_methods import train_step_CRNN_auto_enc, train_step_CRNN, train_step_attention
import os
import time

# type in NN architecture under investigation (this is to save different checkpoints for each architecture)
NN_arch = "transformer"
# choose OS type (linux or windows)
OS = "linux"
# choose whether to train the model or test it
# train ==> "train"
# test ==> "test"
# continue training with previous weights ==> "train_cont"
sim_mode = "train"

normalize_dataset = False
input_window = 1
input_years = 3
input_length, months_per_year, num_windows_per_month = get_input_length(input_window, input_years)
output_length = 30
input_features = 8
output_features = 1
validation_split = 0.2
test_split = 0.1
batch_size = 64
EPOCHS = 700
rnn_units = 128
att_dense_units_1st = 128

region = "Cordoba"
area_code = "06"
series = load_dataset(region, area_code, normalize= normalize_dataset, swap= False, OS = OS)
# randomize the training set to better train the model instead of having nearly similar training examples in each batch
series_shuffled = series.sample(frac = 1)
# split data into training set, development set, test set
train_set, dev_set, test_set = split_dataframe_train_dev_test(series_shuffled, validation_split, test_split)

# create the training set
x_train, x_target_train, y_train = split_sequence(train_set, input_length, output_length, area_code)
# create the dev set
x_dev, x_target_dev, y_dev = split_sequence(dev_set, input_length, output_length, area_code)
# create the test set
x_test, x_target_test, y_test = split_sequence(test_set, input_length, output_length, area_code)

# Initialize optimizer and loss functions
optimizer = tf.keras.optimizers.Adam()
#optimizer = tf.keras.optimizers.Adam( learning_rate= 0.02)
loss_object = tf.keras.losses.MeanSquaredError(reduction='none')

encoder = custom_layers_models.Encoder(rnn_units)
decoder = custom_layers_models.Decoder(rnn_units, output_features, att_dense_units_1st)

# create a checkpoint instance
if (OS == "linux"):
    checkpoint_dir = r"/media/hamamgpu/Drive3/mohamed-hany/cordoba_ckpts"
else:
    checkpoint_dir = r"C:\Users\moham\Desktop\masters\master_thesis\time_series_analysis\model_testing\cordoba checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_" + NN_arch + ".txt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

with tf.device('/gpu:0'):
 if (sim_mode == "train" or sim_mode == "train_cont"):

    if (sim_mode == "train_cont"):
        # Restore the latest checkpoint in checkpoint_dir
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

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
    # train the model using the following gpu
    #with tf.device('/gpu:0'):
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
            output_pred, batch_loss, trainable_variables = train_step_attention(batch_input, batch_output,
                                                                                encoder, decoder, optimizer,
                                                                                loss_object, output_features)

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

            output_dev = evaluate_attention(batch_input, batch_output,
                                            encoder, decoder, output_features)

            #batch_MAE_dev = 0
            #batch_MAPE_dev = 0
            output_dev_pred_array = conv_tensor_array(output_dev)
            # compute MAE & MAPE (per epoch)
            batch_MAE_dev, batch_MAPE_dev = avg_batch_MAE(batch_output, output_dev_pred_array)

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

 else:
    # Restore the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # get number of batches of dev set
    num_batches_test = int(x_test.shape[0] / batch_size)
    remainder_test = x_test.shape[0] % batch_size
    if (remainder_test != 0):
        num_batches_test += 1
    # test the model using the following gpu
    # with tf.device('/gpu:0'):
    test_total_MAE = 0
    test_total_MAPE = 0
    for batch in range(num_batches_test):

        # get the current batch of x_train & y_train
        start_index = batch * batch_size
        end_index = start_index + batch_size
        if (end_index > x_test.shape[0]):
            end_index = x_test.shape[0]
        batch_input = x_test[start_index: end_index]
        batch_target_input = x_target_test[start_index: end_index]
        batch_output = y_test[start_index: end_index]

        output_test = evaluate_attention(batch_input, batch_output,
                                         encoder, decoder, output_features)

        output_test_pred_array = conv_tensor_array(output_test)
        batch_MAE_test, batch_MAPE_test = avg_batch_MAE(batch_output, output_test_pred_array)
        test_total_MAE += batch_MAE_test
        test_total_MAPE += batch_MAPE_test

        print('test_MAE : {} , test_MAPE: {}'.format(test_total_MAE / num_batches_test,
                                                     test_total_MAPE / num_batches_test))

x = 0