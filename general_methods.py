import pandas as pd
import numpy as np
import os
from numpy import array
import tensorflow as tf
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import lstm_att_layers
import seaborn as sns

def load_dataset(region, area_code, normalize = False, swap = True, OS = "linux", linux_path = 0):
    if (OS == "linux"):
        # linux path
        if (linux_path == 1):
            file_path = "/media/hamamgpu/Drive3/mohamed-hany/" + region + ".csv"
        else:
            file_path = "/home/mohamed-hany/Downloads/" + region + ".csv"
    else:
        file_path = "C:\\Users\\moham\\Desktop\\masters\\master_thesis\\time_series_analysis\\data_set\\" + region + ".csv"

    series = pd.read_csv(file_path, index_col=0, parse_dates=True, dayfirst=True)
    series['year'] = series.index.year
    series['month'] = series.index.month
    series['day'] = series.index.day
    series = series.sort_values(by="FECHA")
    #print("initial datatype:\n", series.dtypes)
    #print("initial size:", series["Co06ETo"].size)
    #print("number of rows with n/d:", series.Co06ETo[series.Co06ETo == "n/d"].size)
    #series.Co06ETo[series.Co06ETo == "n/d"] = 0
    #series["Co06ETo"] = pd.to_numeric(series["Co06ETo"], downcast="float")
    #print("current datatype:\n", series.dtypes)
    max_temp = "Co" + area_code + "TMax"
    max_temp_time = "Co" + area_code + "HTMax"
    min_temp = "Co" + area_code + "TMin"
    min_temp_time = "Co" + area_code + "HTMin"
    med_temp = "Co" + area_code + "TMed"
    max_hum = "Co" + area_code + "HumMax"
    min_hum = "Co" + area_code + "HumMin"
    med_hum = "Co" + area_code + "HumMed"
    wind_speed = "Co" + area_code + "VelViento"
    wind_direction = "Co" + area_code + "DirViento"
    radiation = "Co" + area_code + "Rad"
    precipitation = "Co" + area_code + "Precip"
    evapotranspiration = "Co" + area_code + "ETo"


    series.drop(['DIA', precipitation, wind_direction, wind_speed, 'year', 'month', 'day', max_temp_time, min_temp_time],
                axis=1, inplace=True)

    series[max_temp][series[max_temp] == "n/d"] = 0
    series[max_temp] = pd.to_numeric(series[max_temp], downcast="float")
    series[min_temp][series[min_temp] == "n/d"] = 0
    series[min_temp] = pd.to_numeric(series[min_temp], downcast="float")
    series[med_temp][series[med_temp] == "n/d"] = 0
    series[med_temp] = pd.to_numeric(series[med_temp], downcast="float")
    series[max_hum][series[max_hum] == "n/d"] = 0
    series[max_hum] = pd.to_numeric(series[max_hum], downcast="float")
    series[min_hum][series[min_hum] == "n/d"] = 0
    series[min_hum] = pd.to_numeric(series[min_hum], downcast="float")
    series[med_hum][series[med_hum] == "n/d"] = 0
    series[med_hum] = pd.to_numeric(series[med_hum], downcast="float")
    series[radiation][series[radiation] == "n/d"] = 0
    series[radiation] = pd.to_numeric(series[radiation], downcast="float")
    series[evapotranspiration][series[evapotranspiration] == "n/d"] = 0
    series[evapotranspiration] = pd.to_numeric(series[evapotranspiration], downcast="float")

    if normalize == True:
        max_temp_mean = series[max_temp].mean()
        max_temp_stddev = series[max_temp].std()
        series[max_temp] = (series[max_temp] - max_temp_mean)
        series[max_temp] = series[max_temp] / max_temp_stddev

        min_temp_mean = series[min_temp].mean()
        min_temp_stddev = series[min_temp].std()
        series[min_temp] = (series[min_temp] - min_temp_mean)
        series[min_temp] = series[min_temp] / min_temp_stddev

        med_temp_mean = series[med_temp].mean()
        med_temp_stddev = series[med_temp].std()
        series[med_temp] = (series[med_temp] - med_temp_mean)
        series[med_temp] = series[med_temp] / med_temp_stddev

        max_hum_mean = series[max_hum].mean()
        max_hum_stddev = series[max_hum].std()
        series[max_hum] = (series[max_hum] - max_hum_mean)
        series[max_hum] = series[max_hum] / max_hum_stddev

        min_hum_mean = series[min_hum].mean()
        min_hum_stddev = series[min_hum].std()
        series[min_hum] = (series[min_hum] - min_hum_mean)
        series[min_hum] = series[min_hum] / min_hum_stddev

        med_hum_mean = series[med_hum].mean()
        med_hum_stddev = series[med_hum].std()
        series[med_hum] = (series[med_hum] - med_hum_mean)
        series[med_hum] = series[med_hum] / med_hum_stddev

        radiation_mean = series[radiation].mean()
        radiation_stddev = series[radiation].std()
        series[radiation] = (series[radiation] - radiation_mean)
        series[radiation] = series[radiation] / radiation_stddev

        evapotranspiration_mean = series[evapotranspiration].mean()
        evapotranspiration_stddev = series[evapotranspiration].std()
        series[evapotranspiration] = (series[evapotranspiration] - evapotranspiration_mean)
        series[evapotranspiration] = series[evapotranspiration] / evapotranspiration_stddev

    if (swap == True):
        series = swap_dataframe_cols(series, 3, 6)
        series = swap_dataframe_cols(series, 4, 7)


    return series

def swap_dataframe_cols(series, col1_index, col2_index):
    # get a list of the columns
    col_list = list(series)
    # swap columns content
    series[[col_list[col2_index], col_list[col1_index]]] = series[[col_list[col1_index], col_list[col2_index]]]
    # swap column names
    col_list[col2_index], col_list[col1_index] = col_list[col1_index], col_list[col2_index]
    series.columns = col_list
    return series

# split a univariate sequence into samples
def split_sequence_univariate(input_sequence, n_steps_in, n_steps_out):

    x, y = list(), list()
    x_list_arr = list()
    for i in range (len(input_sequence)):
        # find the end of this pattern
        x_end = i + n_steps_in
        y_end = x_end + n_steps_out
        # check if we are beyond the sequence
        if y_end > len(input_sequence):
            break
        # gather input and output parts of the pattern
        seq_x = input_sequence[i : x_end]
        seq_y = input_sequence[x_end : y_end]
        x.append(seq_x)
        y.append(seq_y)
    x_arr = array(x)
    y_arr = array(y)
    return x_arr, y_arr

# split a multivariate sequence into samples
def split_sequence(input_sequence, n_steps_in, n_steps_out, area_code):
    eva_trans = "Co" + area_code + "ETo"
    x, target_x, y = list(), list(), list()
    x_list_arr = list()
    for i in range (len(input_sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(input_sequence):
            break
        # gather input and output parts of the pattern
        seq_x = input_sequence[i : end_ix]
        seq_y = input_sequence[end_ix : out_end_ix]
        seq_y = seq_y[eva_trans]
        seq_x_target = seq_x[eva_trans]
        x.append(seq_x)
        y.append(seq_y)
        target_x.append(seq_x_target)
        x_list_arr.append(array(seq_x))
    target_x_arr, y_arr = array(target_x), array(y)
    #x_arr = array(x)
    x_arr = array(x_list_arr)
    return x_arr, target_x_arr, y_arr

# split a multivariate sequence into samples
def split_sequence_autoenc(input_sequence, n_steps_in, area_code):
    eva_trans = "Co" + area_code + "ETo"
    x, target_x, y = list(), list(), list()
    for i in range (len(input_sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        # check if we are beyond the sequence
        if end_ix > len(input_sequence):
            break
        # gather input and output parts of the pattern
        seq_x = input_sequence[i : end_ix]
        seq_y = input_sequence[i : end_ix]
        seq_y = seq_y[eva_trans]
        seq_x_target = seq_x[eva_trans]
        x.append(seq_x)
        y.append(seq_y)
        target_x.append(seq_x_target)
    return array(x), array(target_x), array(y)

def split_dataframe_train_dev_test(input_series, validation_split, test_split):

    series_size = len(input_series)
    validation_size = int(series_size * validation_split)
    test_size = int(series_size * test_split)
    train_size = int(series_size * (1 - validation_split - test_split))

    train_set = input_series[0: train_size]
    dev_set = input_series[train_size: (train_size + validation_size)]
    test_set = input_series[(train_size + validation_size): (train_size + validation_size + test_size)]

    return train_set, dev_set, test_set

def generate_train_dev_test_sets(series, validation_split, test_split, input_length, output_length, area_code):

    # split data into training set, development set, test set
    train_set, dev_set, test_set = split_dataframe_train_dev_test(series, validation_split, test_split)

    # create the training set
    x_train, x_target_train, y_train = split_sequence(train_set, input_length, output_length, area_code)
    y_train = y_train[..., np.newaxis]

    # shuffle the training set
    shuffler = np.random.permutation(x_train.shape[0])
    x_train = x_train[shuffler]
    x_target_train = x_target_train[shuffler]
    y_train = y_train[shuffler]
    # create the dev set
    x_dev, x_target_dev, y_dev = split_sequence(dev_set, input_length, output_length, area_code)
    y_dev = y_dev[..., np.newaxis]
    # create the test set
    x_test, x_target_test, y_test = split_sequence(test_set, input_length, output_length, area_code)
    y_test = y_test[..., np.newaxis]

    return x_train, y_train, x_dev, y_dev, x_test, y_test

def generate_datasets_univariate(series, validation_split, test_split, input_length, output_length):

    # split data into training set, development set, test set
    train_set, dev_set, test_set = split_dataframe_train_dev_test(series, validation_split, test_split)

    # create the training set
    x_train, y_train = split_sequence_univariate(train_set, input_length, output_length)
    x_train = x_train[..., np.newaxis]
    y_train = y_train[..., np.newaxis]
    # shuffle the training set
    shuffler = np.random.permutation(x_train.shape[0])
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]

    # create the dev set
    x_dev, y_dev = split_sequence_univariate(dev_set, input_length, output_length)
    x_dev = x_dev[..., np.newaxis]
    y_dev = y_dev[..., np.newaxis]

    # create the test set
    x_test, y_test = split_sequence_univariate(test_set, input_length, output_length)
    x_test = x_test[..., np.newaxis]
    y_test = y_test[..., np.newaxis]

    return x_train, y_train, x_dev, y_dev, x_test, y_test

def create_checkpoint(OS, NN_arch, model, optimizer, linux_path = 0):
    # create a checkpoint instance
    if (OS == "linux"):
        if (linux_path == 1):
            checkpoint_dir = r"/media/hamamgpu/Drive3/mohamed-hany/cordoba_ckpts"
        else:
            checkpoint_dir = r"/home/mohamed-hany/Downloads/cordoba_ckpts"
    else:
        checkpoint_dir = r"C:\Users\moham\Desktop\masters\master_thesis\time_series_analysis\model_testing\cordoba checkpoints"

    # checkpoint directory
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_" + NN_arch + ".ckpt")
    # checkpoint object
    checkpoint = tf.train.Checkpoint(model= model, optimizer= optimizer)
    # checkpoint manager
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=5)

    return checkpoint, checkpoint_prefix, ckpt_manager

def get_num_batches(num_examples, batch_size):
    num_batches = int(num_examples / batch_size)
    remainder = num_examples % batch_size
    if (remainder != 0):
        #num_batches += 1
        pass
    return num_batches

def get_batch_data(batch_num, batch_size, x, y):

    # get the current batch data
    start_index = batch_num * batch_size
    end_index = start_index + batch_size
    if (end_index > x.shape[0]):
        end_index = x.shape[0]
    batch_input = x[start_index: end_index]
    #batch_target_input = x_target_train[start_index: end_index]
    batch_output = y[start_index: end_index]

    batch_input = tf.convert_to_tensor(batch_input)
    batch_output = tf.convert_to_tensor(batch_output)

    batch_input = tf.cast(batch_input, tf.float32)
    batch_output = tf.cast(batch_output, tf.float32)

    return batch_input, batch_output

def conv_tensor_array(input_tensor):
    to_list = tf.Variable(input_tensor).numpy().tolist()
    to_array = np.array(to_list)
    # remove dimensions of size 1
    to_array = np.squeeze(to_array)
    # transpose to get desired shape
    to_array = np.transpose(to_array)
    return to_array

def compare_lists(old_list, new_list):

    variables_changed = []
    any_change = False
    for i in range(len(old_list)):
        diff = old_list[i]-new_list[i]
        diff = conv_tensor_array(diff)
        if (diff.ndim > 0):
            sum_diff = sum(diff)
            for j in range(diff.ndim - 1):
                sum_diff = sum(sum_diff)
        else:
            sum_diff = diff
        if (sum_diff != 0):
            #print(f"difference equals {sum_diff}")
            any_change = True
            variables_changed.append(i)
    return any_change, variables_changed

def test_model_w_heatmap(NN_arch, x_test, y_test, input_length, output_length, model, test_accuracy, batch_size,
                         lstm_units):
    x_test_batch = x_test[:batch_size, ...]
    y_test_batch = y_test[:batch_size, ...]
    if (NN_arch == "autoregressive_attention"):
        h_dec_input = tf.Variable(tf.zeros((batch_size, output_length, lstm_units), dtype=tf.float32))
        predicted_output, attention_weights = lstm_att_layers.evaluate(x_test_batch, y_test_batch,
                                                                       input_length, output_length,
                                                                       model, test_accuracy,
                                                                       h_dec_input)
    else:
        predicted_output, attention_weights = lstm_att_layers.evaluate(x_test_batch, y_test_batch,
                                                                       input_length, output_length,
                                                                       model, test_accuracy)

    predicted_output_reshaped = tf.squeeze(tf.convert_to_tensor(predicted_output))
    real_output_reshaped = tf.squeeze(y_test_batch)
    num_dim = np.ndim(real_output_reshaped)
    if (num_dim == 1):
        predicted_output_reshaped = tf.expand_dims(predicted_output_reshaped, axis = -1)
        real_output_reshaped = tf.expand_dims(real_output_reshaped, axis=-1)
    else:
        predicted_output_reshaped = tf.transpose(predicted_output_reshaped)
    real_seq = tf.concat([x_test_batch[..., -1], real_output_reshaped], axis = -1)
    pred_seq = tf.concat([x_test_batch[..., -1], predicted_output_reshaped], axis = -1)
    plt.figure()
    x_plot = np.arange(x_test_batch.shape[1] + y_test_batch.shape[1])
    y_real_plot = real_seq[0, ...]
    y_pred_plot = pred_seq[0, ...]
    plt.plot(x_plot, y_real_plot, 'r', label = "real")
    plt.plot(x_plot, y_pred_plot, 'b', label = "predicted")
    plt.xlabel('time')
    plt.ylabel('evapotranspiration')
    plt.legend()

    plt.show(block = False)

    att_weights_reshaped = tf.squeeze(tf.convert_to_tensor(attention_weights))
    if (num_dim > 1):
        att_weights_reshaped = tf.transpose(att_weights_reshaped, perm = [1, 0, 2])
    else:
        #
        att_weights_reshaped = tf.expand_dims(att_weights_reshaped, axis = 1)
    plt.figure()
    #plt.imshow(att_weights_reshaped[0, ...], cmap='hot', interpolation='nearest')
    sns.heatmap(att_weights_reshaped[0, ...], linewidth=0.5)
    plt.show(block = False)

def block_shuffle_series(series, block_size):
    series_list = series.tolist()
    series_size = len(series_list)
    assert series_size % block_size == 0
    num_blocks = int(series_size / block_size)
    series_block_split = np.array_split(series_list, num_blocks)
    np.random.seed(1)
    np.random.shuffle(series_block_split)
    new_series_list = np.concatenate(series_block_split, axis=0)
    new_series = pd.DataFrame(new_series_list)
    return new_series

def calc_SSE(real, pred):

    real_sq = np.squeeze(real)
    pred_sq = np.squeeze(pred)
    diff = real_sq - pred_sq
    SSE = np.mean(np.square(diff))
    return SSE
    """
    if (np.ndim(real) > 1):
        avg_SSE = np.mean(SSE)
        return avg_SSE
    else:
        return SSE
    """






# Evaluate function -- similar to the training loop
# target is input with shape assuming batch (3-dimensional)
def evaluate(input, target, encoder, decoder):

  # Attention plot (to be plotted later on) -- initialized with max_lengths of both target and input
  attention_plot = np.zeros((input.shape[1], target.shape[1]))

  enc_output, h_enc, c_enc = encoder(input)

  # dec_hidden is used by attention, hence is the same h_enc
  h_dec_prev = h_enc
  c_dec_prev = c_enc

  # first y_prev is all zeros
  y_prev = tf.zeros((target.shape[0], decoder.n_output_features))

  output_pred = []

  for t in range(target.shape[1]):
    # Pass enc_output to the decoder
    y_pred, h_dec, c_dec, attention_weights = decoder(y_prev, h_dec_prev, c_dec_prev, enc_output)

    y_prev = y_pred
    h_dec_prev = h_dec
    c_dec_prev = c_dec
    output_pred.append(y_pred)

    # Store the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[:,t] = attention_weights.numpy()

  return output_pred, attention_plot

def evaluate_autoenc(input, target, encoder, decoder, num_weeks):

    h_enc_0 = tf.zeros((input.shape[0], encoder.RNN_num_units))
    c_enc_0 = tf.zeros((input.shape[0], encoder.RNN_num_units))
    h_enc_prev, c_enc_prev = h_enc_0, c_enc_0
    for i in range(num_weeks):
        start_index = i * 7
        end_index = start_index + 7
        enc_input = input[:, start_index:end_index, :]
        enc_output, h_enc, c_enc = encoder(enc_input, h_enc_prev, c_enc_prev)
        h_enc_prev = h_enc
        c_enc_prev = c_enc

    h_dec_prev = h_enc
    c_dec_prev = c_enc

    # first y_prev is all zeros
    y_prev = tf.zeros((target.shape[0], decoder.n_output_features))
    y_prev = tf.expand_dims(y_prev, axis=1)
    output_pred = []
    output_pred_array = np.zeros((target.shape[0], target.shape[1]))
    for t in range(target.shape[1]):
        # Pass enc_output to the decoder
        y_pred, h_dec, c_dec = decoder(h_dec_prev, c_dec_prev, y_prev)
        y_pred_reshaped = tf.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))
        y_prev = y_pred
        h_dec_prev = h_dec
        c_dec_prev = c_dec
        output_pred.append(y_pred_reshaped)
        y_pred_reshaped = tf.reshape(y_pred_reshaped, (-1))
        output_pred_array[:,t] = y_pred_reshaped
    output_pred_tensor = tf.convert_to_tensor( output_pred)
    MAE, MAPE = avg_batch_MAE(target, output_pred_array)
    return output_pred, MAE, MAPE

# Function for plotting the attention weights
def plot_attention(attention, input, output):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + input, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + output, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show(block=False)

# Translate function (which internally calls the evaluate function)
def forecast(input, target, encoder, decoder):
    output_pred, attention_plot = evaluate(input, target, encoder, decoder)

    target = tf.reshape(target, (-1, ))
    output_pred = tf.reshape(output_pred, (-1, ))
    #error = tf.reshape(target, (-1, )) - tf.reshape(output_pred, (-1, ))
    error = target - output_pred
    error_abs = np.absolute(error)
    MAE = np.mean(error_abs)
    #MAE = np.sum(error_abs, axis = 1)/(target.shape[1])
    fig = plt.figure()
    plt.plot(target, label='ground truth')
    plt.plot(output_pred, label='predicted output')
    plt.legend()
    plt.show(block=False)

    #attention_plot = attention_plot[:len(output_pred.split(' ')), :len(input.split(' '))]
    #plot_attention(attention_plot, input.split(' '), output_pred.split(' '))

    return MAE

def avg_batch_MAE(real, pred):
    MAE = 0
    MAPE = 0
    if (pred.ndim == 1):
        pred_reshaped = np.expand_dims(pred, axis = 0)
    else:
        pred_reshaped = pred
    for i in range(real.shape[0]):
        real_array = real[i]
        pred_array = pred_reshaped[i]
        zero_indices = np.where(real_array==0)[0]
        real_array_wo_zeros = np.delete(real_array,zero_indices)
        pred_array_wo_zeros = np.delete(pred_array, zero_indices)
        if (len(real_array_wo_zeros) != 0):
            MAE += mean_absolute_error(real_array_wo_zeros, pred_array_wo_zeros)
            MAPE += mean_absolute_percentage_error(real_array_wo_zeros, pred_array_wo_zeros) * 100

    batch_MAE = MAE / real.shape[0]
    batch_MAPE = MAPE / real.shape[0]
    #batch_MAE, batch_MAPE = MAE, MAPE
    return batch_MAE, batch_MAPE

def get_resnet_units(input_features, input_window, kernel_size, num_kernels, pool_size):
    conv_pool_out_rows = (input_window - kernel_size + 1) - pool_size + 1
    conv_pool_out_cols = (input_features - kernel_size + 1) - pool_size + 1
    resnet_ffinput_units = conv_pool_out_rows * conv_pool_out_cols * num_kernels
    return resnet_ffinput_units

def get_input_length(input_window, input_years):
    if input_window == 1:
        # assume 1 year has 360 days
        months_per_year = 12
        num_windows_per_month = 30
    if input_window == 7:
        # assume 1 year has 364 days
        months_per_year = 13
        num_windows_per_month = 4
    if input_window == 10:
        # assume 1 year has 360 days
        months_per_year = 12
        num_windows_per_month = 3
    if input_window == 15:
        # assume 1 year has 360 days
        months_per_year = 12
        num_windows_per_month = 2
    if input_window == 30:
        # assume 1 year has 360 days
        months_per_year = 12
        num_windows_per_month = 1
    input_length = int(input_window * num_windows_per_month * months_per_year * input_years)
    return input_length, months_per_year, num_windows_per_month

def plot_pred_vs_target(target, pred):

    plt.plot(pred, color='r', label='prediced output')
    plt.plot(target, color='b', label='real output')
    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    # To load the display window
    plt.show()





