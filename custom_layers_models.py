import tensorflow as tf
import time
import numpy as np
from general_methods import conv_tensor_array

class conv_pool_layer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, num_kernels, pool_size, pool_strides = 1,
                 kernel_regularizer = None, conv_activation = "tanh"):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.kernel_regularizer = kernel_regularizer
        self.conv_activation = conv_activation

        self.conv_layer = tf.keras.layers.Conv2D(filters= num_kernels, kernel_size= kernel_size,
                                                 kernel_regularizer= kernel_regularizer,
                                                 activation= conv_activation)
        self.pooling_layer = tf.keras.layers.MaxPooling2D(pool_size= (pool_size, pool_size), strides= pool_strides)

    def build(self, input_shape):
        pass

    def call(self, input, expand_dims):
        # input shape = (batch_size, timestep_window, input_features)
        if expand_dims == True:
            # make input shape = (batch_size, timestep_window, input_features, 1 channel)
            input = tf.expand_dims(input, axis= -1)

        # conv_output shape = (batch_size, timestep_window - kernel_size + 1,
        # input_features - kernel_size + 1, num_kernels)
        # conv_output shape = (batch_size, conv_out_rows, conv_out_columns, num_kernels)
        conv_output = self.conv_layer(input)

        # pooling_output shape = (batch_size, floor((conv_out_rows - pool_size)/ pool_strides) + 1,
        # floor((conv_out_columns - pool_size)/ pool_strides) + 1, num_kernels)
        pooling_output = self.pooling_layer(conv_output)

        return pooling_output

class conv_pool_1D_layer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, num_kernels, pool_size, pool_strides = 1, kernel_regularizer = None,
                 conv_activation = "tanh"):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.kernel_regularizer = kernel_regularizer
        self.conv_activation = conv_activation

        self.conv_layer = tf.keras.layers.Conv1D(filters= num_kernels, kernel_size= kernel_size,
                                                 kernel_regularizer= kernel_regularizer,
                                                 activation= conv_activation)
        self.pooling_layer = tf.keras.layers.MaxPooling1D(pool_size= pool_size, strides= pool_strides)

    def build(self, input_shape):
        pass

    def call(self, input):
        # input shape = (batch_size, input_units)

        # make input shape = (batch_size, input_units, 1 channel)
        input = tf.expand_dims(input, axis= -1)

        # conv_output shape = (batch_size, input_units - kernel_size + 1, num_kernels)
        # conv_output shape = (batch_size, conv_out_rows, num_kernels)
        conv_output = self.conv_layer(input)

        # pooling_output shape = (batch_size, floor((conv_out_rows - pool_size)/ pool_strides) + 1, num_kernels)
        pooling_output = self.pooling_layer(conv_output)

        return pooling_output

class resnet(tf.keras.layers.Layer):
    def __init__(self, feed_forward_units, Dense_activation = "relu"):
        super().__init__()
        self.feed_forward_units = feed_forward_units

        self.Dense_layer = tf.keras.layers.Dense(feed_forward_units, activation= Dense_activation)

    def build(self, input_shape):
        pass

    def call(self, target_input, feed_forward_input):
        # input shape = (batch_size, timestep_window)
        # feed_forward_input shape = (batch_size, feed_foward_units)

        # Dense_output shape = (batch_size, feed_foward_units)
        Dense_output = self.Dense_layer(target_input)

        #layer_output = Dense_output + feed_forward_input
        layer_output = tf.math.add(Dense_output, feed_forward_input)


        return layer_output

class CRNN_encoder(tf.keras.Model):
    def __init__(self, rnn_units, kernel_size, num_kernels, pool_size, resnet_ffinput_units,
                 pool_strides = 1, resnet_Dense_activation = "relu", kernel_regularizer = None,
                 rnn_regularizer = None, rnn_dropout = 0):
        super().__init__()
        self.rnn_units = rnn_units
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.pool_size = pool_size
        self.resnet_ffinput_units = resnet_ffinput_units
        self.pool_strides = pool_strides
        self.resnet_Dense_activation = resnet_Dense_activation
        self.kernel_regularizer = kernel_regularizer
        self.rnn_regularizer = rnn_regularizer
        self.rnn_dropout = rnn_dropout

        self.conv_pool_layer = conv_pool_layer(kernel_size, num_kernels, pool_size,
                                               pool_strides, kernel_regularizer)
        self.resnet = resnet(resnet_ffinput_units, resnet_Dense_activation)
        self.flatten = tf.keras.layers.Flatten()
        self.lstm = tf.keras.layers.LSTM(rnn_units, return_state= "True", return_sequences= "True",
                                         recurrent_regularizer= rnn_regularizer,
                                         kernel_regularizer= rnn_regularizer,
                                         dropout= rnn_dropout,
                                         recurrent_dropout= rnn_dropout)

    def build(self, input_shape):
        pass

    def call(self, input, target_input, h_prev_lstm, c_prev_lstm):
        # input shape = (batch_size, timestep_window, input_features)
        # target_input shape = (batch_size, timestep_window, 1) #last dimension isn't actually there

        # conv output shape: conv_rows = timestep_window - kernel_size + 1
        #                    conv_cols = input_features - kernel_size + 1
        # conv_pool_output shape = (batch_size, floor((conv_rows - pool_size) / pool_strides) + 1,
        #                           floor((conv_cols - pool_size)/ pool_strides) + 1, num_kernels)
        conv_pool_output = self.conv_pool_layer(input, expand_dims = True)
        conv_pool_output_reshaped = self.flatten(conv_pool_output)
        resnet_output = self.resnet(target_input, conv_pool_output_reshaped)
        resnet_output_reshaped = tf.expand_dims(resnet_output, axis = 1)
        y_lstm, h_lstm, c_lstm = self.lstm(resnet_output_reshaped, initial_state= [h_prev_lstm, c_prev_lstm])

        return y_lstm, h_lstm, c_lstm

class attention_single(tf.keras.layers.Layer):
    def __init__(self, att_dense_units, att_dense_1st, att_dense_2nd,
                 kernel_size = 0, num_kernels = 0, pool_size = 0,
                 conv_activation = "relu", kernel_regularizer = None):
        super().__init__()
        """
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.kernel_regularizer = kernel_regularizer
        self.conv_activation = conv_activation
        self.pool_size = pool_size
        """
        self.att_dense_units = att_dense_units
        self.att_dense_1st = att_dense_1st
        self.att_dense_2nd = att_dense_2nd

        self.Dense_dec = tf.keras.layers.Dense(units= att_dense_units, use_bias= False, activation= None)
        self.Dense_enc = tf.keras.layers.Dense(units= att_dense_units, use_bias= False, activation= None)
        self.Dense_att_1st = tf.keras.layers.Dense(units= att_dense_1st, use_bias= False, activation= "tanh")
        self.Dense_att_2nd = tf.keras.layers.Dense(units= att_dense_2nd, use_bias= False, activation= "relu")
        self.Dense_att_3rd = tf.keras.layers.Dense(units= 1, use_bias= False, activation= "relu")

    def build(self, input_shape):
        pass

    def call(self, h_enc, S_prev_dec):
        # h_enc shape = (batch_size, input_length / timestep_window, enc_lstm_units)
        # S_prev_dec shape = (batch_size, dec_lstm_units)

        # S_prev_repeated shape = (batch_size, input_length / timestep_window, dec_lstm_units)
        S_prev_reshaped = tf.expand_dims(S_prev_dec, axis= 1)
        S_prev_repeated = tf.repeat(S_prev_reshaped, repeats= h_enc.shape[1], axis= 1)

        # apply bahdanau attention first step (additive attention)
        bahdanau_out = self.Dense_enc(h_enc) + self.Dense_dec(S_prev_repeated)
        energies1 = self.Dense_att_1st(bahdanau_out)
        energies2 = self.Dense_att_2nd(energies1)
        energies3 = self.Dense_att_3rd(energies2)
        # attention weights shape : (batch size, input_length / input_window, 1)
        attention_weights = tf.nn.softmax(energies3, axis = 1)

        return bahdanau_out, attention_weights

class CRNN_decoder(tf.keras.Model):
    def __init__(self, RNN_num_units, n_output_features,
                 input_window, input_years, months_per_year, num_windows_per_month,
                 att_single_dense_units, att_single_dense_1st,
                 att_single_dense_2nd,
                 att_mon_year_dense_1st,
                 monthly_kernel_size, monthly_num_kernels, monthly_pool_size,
                 yearly_kernel_size, yearly_num_kernels, yearly_pool_size,
                 output_Dense_activation = "relu"):
        super().__init__()
        self.RNN_num_units = RNN_num_units
        self.output_Dense_activation = output_Dense_activation
        self.n_output_features = n_output_features
        self.input_window = input_window
        self.input_years = input_years
        self.months_per_year = months_per_year
        self.num_windows_per_month = num_windows_per_month
        self.att_single_dense_units = att_single_dense_units
        self.att_single_dense_1st = att_single_dense_1st
        self.att_single_dense_2nd = att_single_dense_2nd
        self.monthly_kernel_size = monthly_kernel_size
        self.monthly_num_kernels = monthly_num_kernels
        self.monthly_pool_size = monthly_pool_size
        self.yearly_kernel_size = yearly_kernel_size
        self.yearly_num_kernels = yearly_num_kernels
        self.yearly_pool_size = yearly_pool_size
        self.att_mon_year_dense_1st = att_mon_year_dense_1st

        # output layer
        self.output_dense = tf.keras.layers.Dense(units= n_output_features,
                                    activation= output_Dense_activation)

        self.lstm = tf.keras.layers.LSTM(self.RNN_num_units,
                                         return_sequences=True,
                                         return_state=True)

        self.att_single_layer = attention_single(att_single_dense_units, att_single_dense_1st, att_single_dense_2nd)
        self.monthly_conv_pool = conv_pool_layer(monthly_kernel_size, monthly_num_kernels, monthly_pool_size)
        self.monthly_conv_pool_2 = conv_pool_layer(monthly_kernel_size, monthly_num_kernels, monthly_pool_size)
        self.flatten = tf.keras.layers.Flatten()
        self.yearly_conv_pool = conv_pool_layer(yearly_kernel_size, yearly_num_kernels, yearly_pool_size)
        self.att_mon_dense1 = tf.keras.layers.Dense(att_mon_year_dense_1st, activation = "tanh")
        self.att_mon_dense2 = tf.keras.layers.Dense(1, activation = "relu")
        self.att_year_dense1 = tf.keras.layers.Dense(att_mon_year_dense_1st, activation = "tanh")
        self.att_year_dense2 = tf.keras.layers.Dense(1, activation = "relu")

    # Encoder network comprises an lstm layer
    def call(self, dec_input, S_prev_dec, C_prev_dec, h_enc):
        bahdanau_out, attention_weights = self.att_single_layer(h_enc, S_prev_dec)
        end_index = 0
        for i in range(self.input_years):
            for j in range(self.months_per_year):
                start_index = end_index
                end_index = start_index + self.num_windows_per_month

                # conv_in shape: (batch size, num_windows_per_month, badhanau_output_units)
                conv_in = bahdanau_out[:, start_index: end_index, :]

                # conv_out shape: (batch size, num_windows_per_month - kernel_size +1,
                # badhanau_output_units - kernel_size + 1, num_kernels)
                monthly_conv_out = self.monthly_conv_pool(conv_in, expand_dims = True)
                conv_out_2nd = self.monthly_conv_pool_2(monthly_conv_out, expand_dims = False)
                conv_flattened = self.flatten(conv_out_2nd)
                monthly_energies1 = self.att_mon_dense1(conv_flattened)
                monthly_energies2 = self.att_mon_dense2(monthly_energies1)

                # monthly_conv_out_per_year shape : (batch size, conv_rows * # months, conv_columns, num_kernels)
                if (j == 0):
                    monthly_conv_out_per_year = monthly_conv_out

                    monthly_attention = monthly_energies2
                    monthly_attention = tf.expand_dims(monthly_attention, axis=1)
                else:
                    monthly_conv_out_per_year = tf.concat([monthly_conv_out_per_year, monthly_conv_out],
                                                          axis = 1)

                    monthly_energies2_reshaped = tf.expand_dims(monthly_energies2, axis=1)
                    monthly_attention = tf.concat([monthly_attention, monthly_energies2_reshaped], axis=1)
            # monthly_attention shape : (batch size, # of months, 1) this is per year
            monthly_attention = tf.nn.softmax(monthly_attention, axis=1)

            yearly_conv_out = self.yearly_conv_pool(monthly_conv_out_per_year, expand_dims = False)
            yearly_conv_out_flattened = self.flatten(yearly_conv_out)
            yearly_energies1 = self.att_year_dense1(yearly_conv_out_flattened)
            yearly_energies2 = self.att_year_dense2(yearly_energies1)

            if (i == 0):
                monthly_attention_for_years = monthly_attention

                yearly_attention = yearly_energies2
                yearly_attention = tf.expand_dims(yearly_attention, axis=1)
            else:
                # monthly_attention_for_years shape : (batch size, # months * # years, 1)
                monthly_attention_for_years = tf.concat([monthly_attention_for_years, monthly_attention], axis = 1)

                yearly_energies2_reshaped = tf.expand_dims(yearly_energies2, axis=1)
                yearly_attention = tf.concat([yearly_attention, yearly_energies2_reshaped], axis = 1)

        # yearly_attention shape : (batch size, number of years, 1)
        yearly_attention = tf.nn.softmax(yearly_attention, axis=1)

        counter = 0
        monthly_att_iterator = 0
        #new_attention_weights = []
        for i in range(self.input_years):
            # for each year use the same yearly attention value
            for j in range(self.months_per_year):
                # for each month use the same monthly attention
                for k in range(self.num_windows_per_month):
                    mul_1 = tf.math.multiply(monthly_attention_for_years[:, monthly_att_iterator, :],
                                             yearly_attention[:, i, :])
                    new_weight = tf.math.multiply(attention_weights[:, counter, :], mul_1)

                    # new_attention_weights shape: (batch size, # years * # months * # iterations per month)
                    if counter == 0:
                        new_attention_weights = new_weight
                    else:
                        new_attention_weights = tf.concat([new_attention_weights, new_weight], axis = -1)

                    counter += 1
                monthly_att_iterator = monthly_att_iterator + 1

        # expand dimensions of new_attention_weights to be : (batch size, # years * # months * # iterations per month, 1)
        # the last added dimensions is used for repetition of the attention weight x times, where x is the hidden units
        # of the encoder output

        new_attention_weights_reshaped = tf.expand_dims(new_attention_weights, axis = -1)
        # repeated shape : (batch size, # years * # months * # iterations per month, hidden size of encoder)
        new_attention_weights_repeated = tf.repeat(new_attention_weights_reshaped, (h_enc.shape.as_list())[-1],
                                               axis=-1)
        # shape = batch_size, # years * # months * # iterations per month, hidden size of encoder
        elementwise_mult = tf.math.multiply(new_attention_weights_repeated, h_enc)

        # sum over the input time steps (weighted average)
        # shape: batch_size, hidden size of encoder
        context_vector = tf.reduce_sum(elementwise_mult, axis=1)

        # new size : (batch size, 1, hidden size of encoder)
        context_vector_reshaped = tf.expand_dims(context_vector, axis = 1)

        new_dec_input = tf.concat([context_vector_reshaped, dec_input], axis = -1)

        # decoder lstm:
        y, h, c = self.lstm(new_dec_input, initial_state= [S_prev_dec, C_prev_dec])
        output = self.output_dense(y)
        return output, h, c, new_attention_weights

class CRNN_auto_decoder(tf.keras.Model):
    def __init__(self, RNN_num_units, n_output_features,
                 Dense_activation = "relu"):
        super().__init__()
        self.RNN_num_units = RNN_num_units
        self.Dense_activation = Dense_activation
        self.n_output_features = n_output_features

        # output layer
        self.Dense = tf.keras.layers.Dense(units= self.n_output_features,
                                    activation= self.Dense_activation)

        self.lstm = tf.keras.layers.LSTM(self.RNN_num_units,
                                         return_sequences=True,
                                         return_state=True)

    # Encoder network comprises an lstm layer
    def call(self, dec_input, dec_h_prev, dec_c_prev):
        y, h, c = self.lstm(dec_input, initial_state= [dec_h_prev, dec_c_prev])
        output = self.Dense(y)
        return output, h, c

# Encoder class
class Encoder(tf.keras.Model):
  def __init__(self, enc_units, batch_size = 0, enc_regularization = None, enc_dropout = 0):
    super().__init__()
    self.batch_size = batch_size
    self.enc_units = enc_units

    # LSTM Layer
    # glorot_uniform: Initializer for the recurrent_kernel weights matrix,
    # used for the linear transformation of the recurrent state
    self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',
                                   recurrent_regularizer = enc_regularization,
                                   kernel_regularizer = enc_regularization,
                                   dropout = enc_dropout,
                                   recurrent_dropout = enc_dropout)

  # Encoder network comprises an lstm layer
  def call(self, input_sequence, hidden = 0):
    y, h_last, c_last = self.lstm(input_sequence)
    return y, h_last, c_last

  # To initialize the hidden state
  #def initialize_hidden_state(self):
    #return tf.zeros((self.enc_units,))

# Attention Mechanism
class Attention_Layer(tf.keras.layers.Layer):
  def __init__(self, units,
               activation1='tanh', activation2 = 'relu'):
    super().__init__()
    self.Dense1 = tf.keras.layers.Dense(units, activation= activation1)
    self.Dense2 = tf.keras.layers.Dense(1, activation= activation2)
    #self.activator = tf.keras.layers.Activation(tf.keras.activations.softmax, name='attention_weights')
    #self.dotor = tf.keras.layers.Dot(axes=1)

  def build(self, input_shape):
    pass

  def call(self, h_encoder, h_decoder_prev):
    # h_encoder shape == (batch_size, input sequence length, hidden size of encoder)
    # h_decoder_prev shape == (batch_size, hidden size of decoder)

    input_sequence_length = h_encoder.shape.as_list()[1]
    # h_decoder_prev with time axis shape == (batch_size, 1, hidden size of decoder)
    h_decoder_prev_w_time_axis = tf.expand_dims(h_decoder_prev, axis=1)

    # h_decoder_prev with time axis shape == (batch_size, input sequence length, hidden size of decoder)
    h_decoder_prev_w_time_axis = tf.repeat(h_decoder_prev_w_time_axis, input_sequence_length, axis=1)

    #concatenate h_decoder_prev with h_encoder
    h_encoder_decoder = tf.concat([h_encoder, h_decoder_prev_w_time_axis], axis = -1)

    # get attention weights (shape = batch_size, input sequence length, 1)
    energies1 = self.Dense1(h_encoder_decoder)
    energies2 = self.Dense2(energies1)
    #attention_weights = self.activator(energies2)
    attention_weights = tf.nn.softmax(energies2, axis = 1)

    # get context vector (shape = batch_size, hidden size of encoder)
    # repeat attention weights so that its shape = batch_size, input sequence length, hidden size of encoder
    # (for dot product calculation)
    attention_weights_repeated = tf.repeat(attention_weights, (h_encoder.shape.as_list())[-1],
                                           axis = -1)
    # shape = batch_size, input sequence length, hidden size of encoder
    elementwise_mult = tf.math.multiply(attention_weights_repeated, h_encoder)

    # sum over the input time steps (weighted average)
    # shape: batch_size, hidden size of encoder
    context_vector = tf.reduce_sum(elementwise_mult, axis = 1)

    return context_vector, attention_weights

# Decoder class
class Decoder(tf.keras.Model):
  def __init__(self, dec_lstm_units, n_output_features,
               attention_units, batch_sz = 0,
               Dense_activation = "relu",
               dec_regularization = None, dec_dropout = 0):
    super().__init__()
    self.batch_sz = batch_sz
    self.dec_lstm_units = dec_lstm_units
    self.n_output_features = n_output_features
    self.lstm = tf.keras.layers.LSTM(self.dec_lstm_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',
                                   recurrent_regularizer = dec_regularization,
                                   kernel_regularizer = dec_regularization,
                                   dropout = dec_dropout,
                                   recurrent_dropout = dec_dropout)
    self.Dense = tf.keras.layers.Dense(n_output_features, activation = Dense_activation)

    # Used for attention
    self.attention = Attention_Layer(attention_units)

  def call(self, y_prev, h_dec_prev, c_dec_prev, h_encoder):
    # y_prev shape == (batch_size, n_output_features)
    # h_dec_prev shape == (batch_size, decoder lstm units)
    # h_encoder shape == (batch_size, input sequence length, encoder units)

    # context_vector shape == (batch_size, encoder units)
    # attention_weights shape == (batch_size, input sequence length, 1)
    context_vector, attention_weights = self.attention(h_encoder, h_dec_prev)

    # x shape after concatenation == (batch_size, 1, encoder units + n_output_features)
    x = tf.concat([tf.expand_dims(context_vector, axis = 1), tf.expand_dims(y_prev, axis = 1)],
                  axis=-1)

    # passing the concatenated vector to the LSTM
    y, h, c = self.lstm(x, initial_state = [h_dec_prev, c_dec_prev])

    # output shape = batch size, 1, n_output_features
    output = self.Dense(y)

    # output shape == (batch_size * 1, n_output_features)
    output = tf.reshape(output, (-1, output.shape[-1]))

    return output, h, c, attention_weights

