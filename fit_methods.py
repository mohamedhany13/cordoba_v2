import tensorflow as tf
import numpy as np
from loss_functions import loss_function
from general_methods import avg_batch_MAE


@tf.function
def train_step_CRNN_auto_enc(inp, targ, encoder, decoder, optimizer, loss_object, encoder_lstm_units,
                             input_window, input_length, max_attention_span):
    # inp should have the evapotranspiration feature in the middle of the matrix not in the last row/column
    # max_attention_span is in years

    # input window is needed to calculate number of encoder lstm iterations
    encoder_num_iterations = int(input_length / input_window)
    # exclude first part of the input that doesn't make a full window
    remainder = input_length % input_window
    if (remainder != 0):
        inp = inp[:, remainder:, :]

    loss = 0
    #target_inp = inp[:, :, inp.shape[2]/2]
    # tf.GradientTape() -- record operations for automatic differentiation
    with tf.GradientTape() as tape:
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
        # Teacher forcing - feeding the target as the next input
        for t in range(targ.shape[1]):

            # Pass enc_output to the decoder
            y_pred, h_dec, c_dec = decoder(y_prev, h_dec_prev, c_dec_prev)
            y_real = tf.expand_dims(targ[:, t], axis = 1)
            y_pred_reshaped = tf.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))
            # Compute the loss
            current_batch_loss = loss_function(y_real, y_pred_reshaped, loss_object)
            loss += current_batch_loss

            # Use teacher forcing
            #y_prev = y_real
            y_prev = y_pred
            h_dec_prev = h_dec
            c_dec_prev = c_dec
            output.append(y_pred_reshaped)

    # compute average loss per time step per batch (this function is executed per batch)
    # (output of loss function is average of the whole batch)
    batch_loss = (loss / int(targ.shape[1]))

    # compute average single training example MAE & MAPE
    #ste_MAE, ste_MAPE = avg_batch_MAE(targ, output)

    # Get the model's variables
    variables = encoder.trainable_variables + decoder.trainable_variables

    # Compute the gradients
    gradients = tape.gradient(loss, variables)

    # Update the variables of the model/network
    optimizer.apply_gradients(zip(gradients, variables))

    #return output, ste_loss, ste_MAE, ste_MAPE, batch_loss, variables
    return output, batch_loss, variables


@tf.function
def train_step_CRNN(inp, target_inp, target_out, encoder, decoder, optimizer, loss_object, encoder_lstm_units,
                             input_window, input_length, output_features):
    # inp should have the evapotranspiration feature in the middle of the matrix not in the last row/column

    # input window is needed to calculate number of encoder lstm iterations
    encoder_num_iterations = int(input_length / input_window)
    # exclude first part of the input that doesn't make a full window
    remainder = input_length % input_window
    if (remainder != 0):
        inp = inp[:, remainder:, :]

    loss = 0
    # tf.GradientTape() -- record operations for automatic differentiation
    with tf.GradientTape() as tape:
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
        # Teacher forcing - feeding the target as the next input
        for t in range(target_out.shape[1]):

            # Pass enc_output to the decoder
            y_pred, S_dec, c_dec, new_attention_weights = decoder(y_prev, S_prev_dec, c_prev_dec, h_enc_arr)
            y_real = tf.expand_dims(target_out[:, t], axis = 1)
            y_pred_reshaped = tf.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))
            # Compute the loss
            current_batch_loss = loss_function(y_real, y_pred_reshaped, loss_object)
            loss += current_batch_loss

            # Use teacher forcing
            #y_prev = y_real
            y_prev = y_pred
            S_prev_dec = S_dec
            c_prev_dec = c_dec
            output.append(y_pred_reshaped)

    # compute average loss per time step per batch (this function is executed per batch)
    # (output of loss function is average of the whole batch)
    batch_loss = (loss / int(target_out.shape[1]))

    # compute average single training example MAE & MAPE
    #ste_MAE, ste_MAPE = avg_batch_MAE(targ, output)

    # Get the model's variables
    variables = encoder.trainable_variables + decoder.trainable_variables

    # Compute the gradients
    gradients = tape.gradient(loss, variables)

    # Update the variables of the model/network
    optimizer.apply_gradients(zip(gradients, variables))

    #return output, ste_loss, ste_MAE, ste_MAPE, batch_loss, variables
    return output, batch_loss, variables

@tf.function
def train_step_attention(inp, target_out, encoder, decoder, optimizer, loss_object, output_features):

    loss = 0
    # tf.GradientTape() -- record operations for automatic differentiation
    with tf.GradientTape() as tape:

        # encoder step

        enc_out, h_enc, c_enc = encoder(inp)

        output = []
        S_prev_dec_0 = tf.zeros((inp.shape[0], decoder.RNN_num_units))
        S_prev_dec = S_prev_dec_0
        c_prev_dec_0 = tf.zeros((inp.shape[0], decoder.RNN_num_units))
        c_prev_dec = c_prev_dec_0
        y_prev_0 = tf.zeros((inp.shape[0], 1, output_features))
        y_prev = y_prev_0
        # Teacher forcing - feeding the target as the next input
        for t in range(target_out.shape[1]):

            # Pass enc_output to the decoder
            y_pred, S_dec, c_dec, attention_weights = decoder(y_prev, S_prev_dec, c_prev_dec, enc_out)
            y_real = tf.expand_dims(target_out[:, t], axis = 1)
            y_pred_reshaped = tf.reshape(y_pred, (y_pred.shape[0], y_pred.shape[-1]))
            # Compute the loss
            current_batch_loss = loss_function(y_real, y_pred_reshaped, loss_object)
            loss += current_batch_loss

            # Use teacher forcing
            #y_prev = y_real
            y_prev = y_pred
            S_prev_dec = S_dec
            c_prev_dec = c_dec
            output.append(y_pred_reshaped)

    # compute average loss per time step per batch (this function is executed per batch)
    # (output of loss function is average of the whole batch)
    batch_loss = (loss / int(target_out.shape[1]))

    # compute average single training example MAE & MAPE
    #ste_MAE, ste_MAPE = avg_batch_MAE(targ, output)

    # Get the model's variables
    variables = encoder.trainable_variables + decoder.trainable_variables

    # Compute the gradients
    gradients = tape.gradient(loss, variables)

    # Update the variables of the model/network
    optimizer.apply_gradients(zip(gradients, variables))

    #return output, ste_loss, ste_MAE, ste_MAPE, batch_loss, variables
    return output, batch_loss, variables