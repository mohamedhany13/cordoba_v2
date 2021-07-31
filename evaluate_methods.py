import tensorflow as tf
from general_methods import avg_batch_MAE, conv_tensor_array
import matplotlib.pyplot as plt

def evaluate_CRNN_auto_enc(inp, targ, encoder, decoder, encoder_lstm_units,
                             input_window, input_length, max_attention_span):


    # inp should have the evapotranspiration feature in the middle of the matrix not in the last row/column
    # max_attention_span is in years

    # input window is needed to calculate number of encoder lstm iterations
    encoder_num_iterations = int(input_length / input_window)
    # exclude first part of the input that doesn't make a full window
    remainder = input_length % input_window
    if (remainder != 0):
        inp = inp[:, remainder:, :]

    h_enc_0= tf.zeros((inp.shape[0],encoder_lstm_units))
    c_enc_0= tf.zeros((inp.shape[0],encoder_lstm_units))
    h_enc_prev, c_enc_prev = h_enc_0, c_enc_0
    for i in range(encoder_num_iterations):
        start_index = i * input_window
        end_index = start_index + input_window
        input = inp[:,start_index:end_index,:]
        target_input = targ[:,start_index:end_index]
        enc_output, h_enc, c_enc = encoder(input, target_input, h_enc_prev, c_enc_prev)
        h_enc_prev = h_enc
        c_enc_prev = c_enc

    h_dec_prev = h_enc
    c_dec_prev = c_enc

    # first y_prev is all zeros
    y_prev = tf.zeros((targ.shape[0], decoder.n_output_features))
    y_prev = tf.expand_dims(y_prev, axis = 1)
    output = []

    for t in range(targ.shape[1]):

        # Pass enc_output to the decoder
        y_pred, h_dec, c_dec = decoder(y_prev, h_dec_prev, c_dec_prev)
        #y_real = tf.expand_dims(targ[:, t], axis = 1)
        y_pred_reshaped = tf.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))

        y_prev = y_pred
        h_dec_prev = h_dec
        c_dec_prev = c_dec
        output.append(y_pred_reshaped)

    output_pred_array = conv_tensor_array(output)
    batch_MAE, batch_MAPE = avg_batch_MAE(targ, output_pred_array)

    return output_pred_array, batch_MAE, batch_MAPE

def forecast_CRNN_auto_enc(inp, targ, encoder, decoder, encoder_lstm_units,
                             input_window, input_length, max_attention_span):


    # inp shape = (Tx, input features)
    # targ shape = (Tx, output features)

    # change inp shape to : (1, Tx, input features)
    input_reshaped = tf.expand_dims(inp, axis = 0)
    # change targ shape to : (1, Tx, output features)
    target_reshaped = tf.expand_dims(targ, axis = 0)
    #input_reshaped = inp
    #target_reshaped = targ

    # input window is needed to calculate number of encoder lstm iterations
    encoder_num_iterations = int(input_length / input_window)
    # exclude first part of the input that doesn't make a full window
    remainder = input_length % input_window
    if (remainder != 0):
        input_reshaped = input_reshaped[:, remainder:, :]

    h_enc_0= tf.zeros((input_reshaped.shape[0],encoder_lstm_units))
    c_enc_0= tf.zeros((input_reshaped.shape[0],encoder_lstm_units))
    h_enc_prev, c_enc_prev = h_enc_0, c_enc_0
    for i in range(encoder_num_iterations):
        start_index = i * input_window
        end_index = start_index + input_window
        input = input_reshaped[:,start_index:end_index,:]
        target_input = target_reshaped[:,start_index:end_index]
        enc_output, h_enc, c_enc = encoder(input, target_input, h_enc_prev, c_enc_prev)
        h_enc_prev = h_enc
        c_enc_prev = c_enc

    h_dec_prev = h_enc
    c_dec_prev = c_enc

    # first y_prev is all zeros
    y_prev = tf.zeros((target_reshaped.shape[0], decoder.n_output_features))
    y_prev = tf.expand_dims(y_prev, axis = 1)
    output = []

    for t in range(target_reshaped.shape[1]):

        # Pass enc_output to the decoder
        y_pred, h_dec, c_dec = decoder(y_prev, h_dec_prev, c_dec_prev)
        #y_real = tf.expand_dims(targ[:, t], axis = 1)
        y_pred_reshaped = tf.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))

        y_prev = y_pred
        h_dec_prev = h_dec
        c_dec_prev = c_dec
        output.append(y_pred_reshaped)

    output_pred_array = conv_tensor_array(output)
    predicted_APE = []
    for i in range(targ.shape[0]):
        if (targ[i] != 0):
            APE = (abs(targ[i] - output_pred_array[i])/targ[i]) * 100
            predicted_APE.append(APE)

    plt.plot(output_pred_array, color='r', label='prediced output')
    plt.plot(targ, color='b', label='real output')
    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()
    # To load the display window
    plt.show()

    return output_pred_array, predicted_APE

def evaluate_CRNN(inp, target_inp, target_out, encoder, decoder, encoder_lstm_units,
                             input_window, input_length, output_features):
    # inp should have the evapotranspiration feature in the middle of the matrix not in the last row/column
    # input window is needed to calculate number of encoder lstm iterations
    encoder_num_iterations = int(input_length / input_window)
    # exclude first part of the input that doesn't make a full window
    remainder = input_length % input_window
    if (remainder != 0):
        inp = inp[:, remainder:, :]

    h_enc_0= tf.zeros((inp.shape[0],encoder_lstm_units))
    c_enc_0= tf.zeros((inp.shape[0],encoder_lstm_units))
    h_enc_prev, c_enc_prev = h_enc_0, c_enc_0
    for i in range(encoder_num_iterations):
        start_index = i * input_window
        end_index = start_index + input_window
        input = inp[:,start_index:end_index,:]
        target_input = target_inp[:,start_index:end_index]
        enc_output, h_enc, c_enc = encoder(input, target_input, h_enc_prev, c_enc_prev)
        h_enc_prev = h_enc
        c_enc_prev = c_enc

        if (i == 0):
            h_enc_arr = h_enc
            h_enc_arr = tf.expand_dims(h_enc_arr, axis = 1)
        else:
            h_enc_reshaped = tf.expand_dims(h_enc, axis = 1)
            h_enc_arr = tf.concat([h_enc_arr, h_enc_reshaped], axis = 1)

    h_dec_prev = h_enc
    c_dec_prev = c_enc

    output = []
    S_prev_dec_0 = tf.zeros((inp.shape[0], decoder.RNN_num_units))
    S_prev_dec = S_prev_dec_0
    c_prev_dec_0 = tf.zeros((inp.shape[0], decoder.RNN_num_units))
    c_prev_dec = c_prev_dec_0
    y_prev_0 = tf.zeros((inp.shape[0], 1, output_features))
    y_prev = y_prev_0

    for t in range(target_out.shape[1]):

        # Pass enc_output to the decoder
        y_pred, S_dec, c_dec, new_attention_weights = decoder(y_prev, S_prev_dec, c_prev_dec, h_enc_arr)
        y_pred_reshaped = tf.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))

        y_prev = y_pred
        S_prev_dec = S_dec
        c_prev_dec = c_dec
        output.append(y_pred_reshaped)

    output_pred_array = conv_tensor_array(output)
    batch_MAE, batch_MAPE = avg_batch_MAE(target_out, output_pred_array)

    return output_pred_array, batch_MAE, batch_MAPE

@tf.function
def evaluate_attention(inp, target_out, encoder, decoder, output_features):

    enc_out, h_enc, c_enc = encoder(inp)

    output = []
    S_prev_dec_0 = tf.zeros((inp.shape[0], decoder.RNN_num_units))
    S_prev_dec = S_prev_dec_0
    c_prev_dec_0 = tf.zeros((inp.shape[0], decoder.RNN_num_units))
    c_prev_dec = c_prev_dec_0
    y_prev_0 = tf.zeros((inp.shape[0], output_features))
    y_prev = y_prev_0

    for t in range(target_out.shape[1]):

        # Pass enc_output to the decoder
        y_pred, S_dec, c_dec, new_attention_weights = decoder(y_prev, S_prev_dec, c_prev_dec, enc_out)
        y_pred_reshaped = tf.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))

        y_prev = y_pred
        S_prev_dec = S_dec
        c_prev_dec = c_dec
        output.append(y_pred_reshaped)

    #output_pred_array = conv_tensor_array(output)
    #batch_MAE, batch_MAPE = avg_batch_MAE(target_out, output_pred_array)

    #return output_pred_array, batch_MAE, batch_MAPE
    return output