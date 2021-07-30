import pandas as pd
import numpy as np
from numpy import array
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def load_dataset(region, area_code, normalize = False, swap = True):
    file_path = "/media/hamamgpu/Drive3/mohamed-hany/" + region + ".csv"
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


# split a multivariate sequence into samples
def split_sequence(input_sequence, n_steps_in, n_steps_out, area_code):
    eva_trans = "Co" + area_code + "ETo"
    x, target_x, y = list(), list(), list()
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
    return array(x), array(target_x), array(y)

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


# Loss function
def loss_function(real, pred, loss_object):

  # If there's a '0' in the sequence, the loss is being nullified
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  #loss_ *= mask

  return tf.reduce_mean(loss_)
# mean batch loss?


@tf.function
def train_step(inp, targ, encoder, decoder, optimizer, loss_object):
  loss = 0

  # tf.GradientTape() -- record operations for automatic differentiation
  with tf.GradientTape() as tape:
    enc_output, h_enc, c_enc = encoder(inp)

    h_dec_prev = h_enc
    c_dec_prev = c_enc

    # first y_prev is all zeros
    y_prev = tf.zeros((targ.shape[0], decoder.n_output_features))
    #y_prev = tf.expand_dims(y_prev, axis = 1)
    output = []
    # Teacher forcing - feeding the target as the next input
    for t in range(targ.shape[1]):

      # Pass enc_output to the decoder
      y_pred, h_dec, c_dec, attention_weights = decoder(y_prev, h_dec_prev, c_dec_prev, enc_output)
      y_real = tf.expand_dims(targ[:, t], axis = 1)
      # Compute the loss
      loss += loss_function(y_real, y_pred, loss_object)

      # Use teacher forcing
      y_prev = y_real
      h_dec_prev = h_dec
      c_dec_prev = c_dec
      output.append(y_pred)

  # As this function is called per batch, compute the batch_loss
  batch_loss = (loss / int(targ.shape[1]))

  # Get the model's variables
  variables = encoder.trainable_variables + decoder.trainable_variables

  # Compute the gradients
  gradients = tape.gradient(loss, variables)

  # Update the variables of the model/network
  optimizer.apply_gradients(zip(gradients, variables))

  return output, batch_loss, variables

@tf.function
def train_step(inp, targ, encoder, decoder, optimizer, loss_object):
  loss = 0

  # tf.GradientTape() -- record operations for automatic differentiation
  with tf.GradientTape() as tape:
    enc_output, h_enc, c_enc = encoder(inp)

    h_dec_prev = h_enc
    c_dec_prev = c_enc

    # first y_prev is all zeros
    y_prev = tf.zeros((targ.shape[0], decoder.n_output_features))
    #y_prev = tf.expand_dims(y_prev, axis = 1)
    output = []
    # Teacher forcing - feeding the target as the next input
    for t in range(targ.shape[1]):

      # Pass enc_output to the decoder
      y_pred, h_dec, c_dec, attention_weights = decoder(y_prev, h_dec_prev, c_dec_prev, enc_output)
      y_real = tf.expand_dims(targ[:, t], axis = 1)
      # Compute the loss
      loss += loss_function(y_real, y_pred, loss_object)

      # Use teacher forcing
      y_prev = y_real
      h_dec_prev = h_dec
      c_dec_prev = c_dec
      output.append(y_pred)

  # As this function is called per batch, compute the batch_loss
  batch_loss = (loss / int(targ.shape[1]))

  # Get the model's variables
  variables = encoder.trainable_variables + decoder.trainable_variables

  # Compute the gradients
  gradients = tape.gradient(loss, variables)

  # Update the variables of the model/network
  optimizer.apply_gradients(zip(gradients, variables))

  return output, batch_loss, variables

@tf.function
def train_step_v1(inp, targ, encoder, decoder, optimizer, loss_object, num_weeks, encoder_units):
  loss = 0

  # tf.GradientTape() -- record operations for automatic differentiation
  with tf.GradientTape() as tape:
    h_enc_0= tf.zeros((inp.shape[0],encoder_units))
    c_enc_0= tf.zeros((inp.shape[0],encoder_units))
    h_enc_prev, c_enc_prev = h_enc_0, c_enc_0
    for i in range(num_weeks):
        start_index = i * 7
        end_index = start_index + 7
        input = inp[:,start_index:end_index,:]
        enc_output, h_enc, c_enc = encoder(input, h_enc_prev, c_enc_prev)
        h_enc_prev = h_enc
        c_enc_prev = c_enc

    h_dec_prev = h_enc
    c_dec_prev = c_enc

    # first y_prev is all zeros
    y_prev = tf.zeros((targ.shape[0], decoder.n_output_features))
    y_prev = tf.expand_dims(y_prev, axis = 1)
    output = []
    # Teacher forcing - feeding the target as the next input
    for t in range(targ.shape[1]):

      # Pass enc_output to the decoder
      y_pred, h_dec, c_dec = decoder(h_dec_prev, c_dec_prev, y_prev)
      y_real = tf.expand_dims(targ[:, t], axis = 1)
      y_pred_reshaped = tf.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))
      # Compute the loss
      loss += loss_function(y_real, y_pred_reshaped, loss_object)

      # Use teacher forcing
      #y_prev = y_real
      y_prev = y_pred
      h_dec_prev = h_dec
      c_dec_prev = c_dec
      output.append(y_pred_reshaped)

  # As this function is called per batch, compute the batch_loss
  batch_loss = (loss / int(targ.shape[1]))

  # Get the model's variables
  variables = encoder.trainable_variables + decoder.trainable_variables

  # Compute the gradients
  gradients = tape.gradient(loss, variables)

  # Update the variables of the model/network
  optimizer.apply_gradients(zip(gradients, variables))

  return output, batch_loss, variables


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

def conv_tensor_array(input_tensor):
    to_list = tf.Variable(input_tensor).numpy().tolist()
    to_array = np.array(to_list)
    # remove dimensions of size 1
    to_array = np.squeeze(to_array)
    # transpose to get desired shape
    to_array = np.transpose(to_array)
    return to_array

def split_dataframe_train_dev_test(input_series, validation_split, test_split):

    series_size = len(input_series)
    validation_size = int(series_size * validation_split)
    test_size = int(series_size * test_split)
    train_size = int(series_size * (1 - validation_split - test_split))

    train_set = input_series[0: train_size]
    dev_set = input_series[train_size: (train_size + validation_size)]
    test_set = input_series[(train_size + validation_size): (train_size + validation_size + test_size)]

    return train_set, dev_set, test_set

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




