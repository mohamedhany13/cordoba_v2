import tensorflow as tf
import logging
import numpy as np
import general_methods
import time

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


def create_look_ahead_mask(Tx, Ty, hidden_size, current_time_step):
    # current_time_step should be from start of decoder sequence (not start of whole sequence)
    ones = tf.ones((Tx + current_time_step, hidden_size))
    zeros = tf.zeros((Ty - current_time_step, hidden_size))
    mask = tf.concat([ones, zeros], axis = 0)
    return mask  # (Tx+Ty, hidden size)

class pointwise_attention(tf.keras.layers.Layer):
  def __init__(self, num_units,
               activation1='tanh', activation2 = 'relu'):
    super().__init__()
    self.Dense1 = tf.keras.layers.Dense(num_units, activation= activation1)
    self.Dense2 = tf.keras.layers.Dense(1, activation= activation2)

  def call(self, h_enc, h_dec_prev, Tx, Ty):
    # h_enc shape == (batch_size, input sequence length, hidden size of encoder)
    # h_dec_prev shape == (batch_size, hidden size of decoder)

    input_sequence_length = h_enc.shape[1]
    # h_dec_prev with time axis shape == (batch_size, 1, hidden size of decoder)
    h_dec_prev_w_time_axis = tf.expand_dims(h_dec_prev, axis=1)

    # h_decoder_prev with time axis shape == (batch_size, input sequence length, hidden size of decoder)
    #h_decoder_prev_w_time_axis = tf.repeat(h_dec_prev_w_time_axis, input_sequence_length, axis=1)
    h_decoder_prev_w_time_axis = tf.tile(h_dec_prev_w_time_axis, [1, input_sequence_length, 1])

    #concatenate h_decoder_prev with h_encoder
    h_encoder_decoder = tf.concat([h_enc, h_decoder_prev_w_time_axis], axis = -1)

    # get attention weights (shape = batch_size, input sequence length, 1)
    energies1 = self.Dense1(h_encoder_decoder)
    energies2 = self.Dense2(energies1)
    #attention_weights = self.activator(energies2)
    attention_weights = tf.nn.softmax(energies2, axis = 1)

    # get context vector (shape = batch_size, hidden size of encoder)
    # repeat attention weights so that its shape = batch_size, input sequence length, hidden size of encoder
    # (for dot product calculation)
    #attention_weights_repeated = tf.repeat(attention_weights, h_enc.shape[-1], axis = -1)
    attention_weights_repeated = tf.tile(attention_weights, [1, 1, h_enc.shape[-1]])
    # shape = batch_size, input sequence length, hidden size of encoder
    elementwise_mult = tf.math.multiply(attention_weights_repeated, h_enc)

    # sum over the input time steps (weighted average)
    # shape: batch_size, hidden size of encoder
    context_vector = tf.reduce_sum(elementwise_mult, axis = 1)

    return context_vector, attention_weights

class autoregressive_attention(tf.keras.layers.Layer):
  def __init__(self, num_units,
               activation1='tanh', activation2 = 'relu'):
    super().__init__()
    self.Dense1 = tf.keras.layers.Dense(num_units, activation= activation1)
    self.Dense2 = tf.keras.layers.Dense(1, activation= activation2)

  def call(self, h_enc, h_dec, h_dec_prev, Tx, Ty, current_time_step):
    # h_enc shape == (batch_size, input sequence length, hidden size of encoder)
    # h_dec shape == (batch_size, output sequence length, hidden size of decoder)
    # hidden size of decoder = hidden size of encoder
    # current_time_step should be from start of decoder sequence (not start of whole sequence)

    #h_dec_prev = h_dec[:, current_time_step - 1, :] # doesn't handle the case when current time step = 0
    # h_dec_prev with time axis shape == (batch_size, 1, hidden size of decoder)
    h_dec_prev_w_time_axis = tf.expand_dims(h_dec_prev, axis=1)

    # h_decoder_prev with time axis shape == (batch_size, input sequence length, hidden size of decoder)
    #h_dec_prev_w_t_repeated = tf.repeat(h_dec_prev_w_time_axis, Tx + Ty, axis=1)
    h_dec_prev_w_t_repeated = tf.tile(h_dec_prev_w_time_axis, [1, Tx + Ty, 1])

    #concatenate h_dec with h_enc
    h_enc_dec = tf.concat([h_enc, h_dec], axis = 1)

    # add h_enc_dec with h_dec_prev_w_t_repeated
    h_enc_dec_att = tf.math.add(h_enc_dec, h_dec_prev_w_t_repeated)

    look_ahead_mask_1 = create_look_ahead_mask(Tx, Ty, h_enc_dec_att.shape[-1], current_time_step)
    # zero out future hidden states + current hidden state
    h_enc_dec_att_tuned = tf.math.multiply(h_enc_dec_att, look_ahead_mask_1)

    # get attention weights (shape = batch_size, input sequence length, 1)
    energies1 = self.Dense1(h_enc_dec_att_tuned)
    look_ahead_mask_2 = create_look_ahead_mask(Tx, Ty, tf.shape(energies1)[-1], current_time_step)
    energies1_tuned = tf.math.multiply(energies1, look_ahead_mask_2)
    energies2 = self.Dense2(energies1_tuned)
    look_ahead_mask_3 = create_look_ahead_mask(Tx, Ty, tf.shape(energies2)[-1], current_time_step)
    look_ahead_mask_3 = tf.cast(look_ahead_mask_3, tf.bool)
    look_ahead_mask_3_inv = tf.logical_not(look_ahead_mask_3)
    look_ahead_mask_3_inv = tf.cast(look_ahead_mask_3_inv, tf.float32)
    energies2 += (look_ahead_mask_3_inv * -1e9)

    # attention_weights shape : (batch size, Tx + Ty, 1)
    attention_weights = tf.nn.softmax(energies2, axis = 1)

    # get context vector (shape = batch_size, hidden size of encoder)
    # repeat attention weights so that its shape = batch_size, input + output sequence length, hidden size of encoder
    # (for dot product calculation)
    #attention_weights_repeated = tf.repeat(attention_weights, tf.shape(h_enc_dec)[-1], axis = -1)
    attention_weights_repeated = tf.tile(attention_weights, [1, 1, tf.shape(h_enc_dec)[-1]])
    # shape = batch_size, input sequence length, hidden size of encoder
    elementwise_mult = tf.math.multiply(attention_weights_repeated, h_enc_dec)

    # sum over the input time steps (weighted average)
    # shape: batch_size, hidden size of encoder
    context_vector = tf.reduce_sum(elementwise_mult, axis = 1)

    return context_vector, attention_weights

class autoreg_att_model(tf.keras.Model):
    def __init__(self, output_features, enc_hidden_units, att_hidden_units, enc_dropout):
        super().__init__()

        self.enc_dec_hidden_units = enc_hidden_units
        self.att_hidden_units = att_hidden_units
        self.output_features = output_features

        self.enc_lstm = tf.keras.layers.LSTM(enc_hidden_units, dropout=enc_dropout, recurrent_dropout=enc_dropout,
                                             return_state= True, return_sequences= True)
        self.dec_lstm = tf.keras.layers.LSTM(enc_hidden_units, dropout=enc_dropout, recurrent_dropout=enc_dropout,
                                             return_state= True, return_sequences= True)
        self.auto_reg_att = autoregressive_attention(att_hidden_units)
        #self.pointwise_att = pointwise_attention(att_hidden_units)
        self.final_layer = tf.keras.layers.Dense(output_features)

    def call(self, input_seq, Tx, Ty, h_dec_input):

        y_pred = []
        att_weights = []
        # encoder:
        y_enc, h_enc, c_enc = self.enc_lstm(input_seq, initial_state= None)

        for i in range(Ty):
            if (i == 0):
                h_dec_prev = h_enc
                c_dec_prev = c_enc
                y_dec_prev = input_seq[:, -1, -1]
                y_dec_prev = tf.expand_dims(y_dec_prev, axis = -1)
            else:
                h_dec_prev = h_dec_current
                c_dec_prev = c_dec_current
                y_dec_prev = y_final_current
                y_dec_prev = y_dec_prev[..., -1]

            context_vector, att_weights_single = self.auto_reg_att(y_enc, h_dec_input, h_dec_prev,
                                                            Tx= input_seq.shape[1], Ty= Ty, current_time_step = i)
            #context_vector, att_weights_single = self.pointwise_att(y_enc, h_dec_prev, Tx, Ty)
            x_dec_current = tf.concat([y_dec_prev, context_vector], axis = -1)
            x_dec_current = tf.expand_dims(x_dec_current, axis = 1)
            y_dec_current, h_dec_current, c_dec_current = self.dec_lstm(x_dec_current,
                                                                        initial_state=[h_dec_prev, c_dec_prev])
            y_final_current = self.final_layer(y_dec_current)

            y_pred.append(y_final_current)
            att_weights.append(att_weights_single)
            h_dec_input[:, i, :].assign(h_dec_current)

        return y_pred, att_weights

def loss_function(real, pred, loss_object):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  if (mask.shape != loss_.shape):
    mask = tf.squeeze(mask)
    loss_ = tf.squeeze(loss_)

  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

def accuracy_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  mask = tf.cast(mask, dtype=tf.float32)
  error = tf.math.subtract(real, pred)
  error = tf.math.multiply(error, mask)

  # absolute error
  abs_error = tf.math.abs(error)

  # absolute percentage error
  percentage_error = tf.math.divide_no_nan(error, real) * 100
  abs_percentage_error = tf.math.abs(percentage_error)

  # squared error
  squared_error = tf.math.square(error)

  abs_error = tf.cast(abs_error, dtype=tf.float32)
  abs_percentage_error = tf.cast(abs_percentage_error, dtype=tf.float32)
  squared_error = tf.cast(squared_error, dtype=tf.float32)
  #mask = tf.cast(mask, dtype=tf.float32)

  abs_error_per_timestep = tf.reduce_sum(abs_error)/tf.reduce_sum(mask)
  abs_percentage_error_per_timestep = tf.reduce_sum(abs_percentage_error)/tf.reduce_sum(mask)
  squared_error_per_timestep = tf.reduce_sum(squared_error)/tf.reduce_sum(mask)
  return abs_error_per_timestep, abs_percentage_error_per_timestep, squared_error_per_timestep

class enc_dec_model(tf.keras.Model):
    def __init__(self, output_features, enc_hidden_units, att_hidden_units, enc_dropout):
        super().__init__()

        self.enc_dec_hidden_units = enc_hidden_units
        self.att_hidden_units = att_hidden_units
        self.output_features = output_features

        self.y_pred = None
        self.att_weights = None
        self.h_enc = None
        self.h_dec = None

        self.enc_lstm = tf.keras.layers.LSTM(enc_hidden_units, dropout=enc_dropout, recurrent_dropout=enc_dropout,
                                             return_state= True, return_sequences= True)
        self.dec_lstm = tf.keras.layers.LSTM(enc_hidden_units, dropout=enc_dropout, recurrent_dropout=enc_dropout,
                                             return_state= True, return_sequences= True)
        self.final_layer = tf.keras.layers.Dense(output_features)

    def call(self, input_seq, Tx, Ty):

        # encoder:
        y_enc, h_enc, c_enc = self.enc_lstm(input_seq, initial_state= None)

        y_dec_prev = input_seq[:, -1, -1]
        y_dec_prev = tf.expand_dims(tf.expand_dims(y_dec_prev, axis = -1), axis = -1)

        y_dec_current, h_dec_current, c_dec_current = self.dec_lstm(y_dec_prev, initial_state=[h_enc, c_enc])

        y_final_current = self.final_layer(y_dec_current)

        return y_final_current, 0

class attention_model(tf.keras.Model):
    def __init__(self, output_features, enc_hidden_units, att_hidden_units, enc_dropout):
        super().__init__()

        self.enc_dec_hidden_units = enc_hidden_units
        self.att_hidden_units = att_hidden_units
        self.output_features = output_features

        self.y_pred = None
        self.att_weights = None
        self.h_enc = None
        self.h_dec = None

        self.enc_lstm = tf.keras.layers.LSTM(enc_hidden_units, dropout=enc_dropout, recurrent_dropout=enc_dropout,
                                             return_state= True, return_sequences= True)
        self.dec_lstm = tf.keras.layers.LSTM(enc_hidden_units, dropout=enc_dropout, recurrent_dropout=enc_dropout,
                                             return_state= True, return_sequences= True)
        self.pointwise_att = pointwise_attention(att_hidden_units)
        self.final_layer = tf.keras.layers.Dense(output_features)

    def call(self, input_seq, Tx, Ty):

        # encoder:
        y_enc, h_enc, c_enc = self.enc_lstm(input_seq, initial_state= None)

        for i in range(Ty):
            if (i == 0):
                h_dec_prev = h_enc
                c_dec_prev = c_enc
                y_prev = input_seq[:, -1, -1]
                y_prev = tf.expand_dims(y_prev, axis = -1)
            else:
                h_dec_prev = h_dec_current
                c_dec_prev = c_dec_current
                y_prev = y_final_current
                y_prev = tf.expand_dims(y_prev, axis = -1)
            context_vector, attention_weights = self.pointwise_att(y_enc, h_dec_prev, Tx, Ty)
            x_dec = tf.concat([y_prev, context_vector], axis = -1)
            x_dec = y_prev = tf.expand_dims(x_dec, axis = 1)
            y_dec_current, h_dec_current, c_dec_current = self.dec_lstm(x_dec, initial_state=[h_dec_prev, c_dec_prev])

            y_final_current = self.final_layer(y_dec_current)

        return y_final_current, attention_weights

class lstm_model(tf.keras.Model):
    def __init__(self, enc_hidden_units, output_features):
        super().__init__()

        self.enc_dec_hidden_units = enc_hidden_units
        self.output_features = output_features

        self.enc_lstm = tf.keras.layers.LSTM(enc_hidden_units, return_state= False, return_sequences= False)
        self.final_layer = tf.keras.layers.Dense(output_features, activation= "relu")

    def call(self, input_seq):
        # encoder:
        lstm_out = self.enc_lstm(input_seq)
        y_pred = self.final_layer(lstm_out)

        return y_pred


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
"""
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
@tf.function(input_signature=train_step_signature)
"""

@tf.function
def train_step_lstm(input_seq, target, Tx, Ty, model, optimizer, train_loss, train_accuracy, loss_object):

    with tf.GradientTape() as tape:

        predictions = model(input_seq)
        loss = loss_function(target, predictions, loss_object)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    AE, APE, SE = accuracy_function(target, predictions)

    train_loss(loss)
    train_accuracy(APE)

    return predictions, model.trainable_variables

@tf.function
def evaluate_lstm(input_seq, target, Tx, Ty, model, dev_accuracy):
    predictions = model(input_seq)
    AE, APE, SE = accuracy_function(target, predictions)
    dev_accuracy(APE)

    return predictions

@tf.function
def train_step(input_seq, target, Tx, Ty, autoreg_att_model, optimizer, train_loss, train_accuracy, loss_object,
               h_dec_input = None):

    with tf.GradientTape() as tape:
        if (h_dec_input is None):
            predictions, attention_weights = autoreg_att_model(input_seq, Tx, Ty)
        else:
            predictions, attention_weights = autoreg_att_model(input_seq, Tx, Ty, h_dec_input)
        loss = loss_function(target, predictions, loss_object)

    gradients = tape.gradient(loss, autoreg_att_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoreg_att_model.trainable_variables))

    AE, APE, SE = accuracy_function(target, predictions)

    train_loss(loss)
    train_accuracy(APE)

    return predictions, attention_weights, autoreg_att_model.trainable_variables

@tf.function
def evaluate(input_seq, target, Tx, Ty, autoreg_att_model, dev_accuracy, h_dec_input = None):

    if (h_dec_input is None):
        predictions, attention_weights = autoreg_att_model(input_seq, Tx, Ty)
    else:
        predictions, attention_weights = autoreg_att_model(input_seq, Tx, Ty, h_dec_input)

    AE, APE, SE = accuracy_function(target, predictions)

    dev_accuracy(APE)

    return predictions, attention_weights

def training_function_lstm(sim_mode, checkpoint, checkpoint_prefix, ckpt_manager, batch_size, EPOCHS, train_loss,
                           optimizer, loss_object, train_accuracy, dev_accuracy, model,
                           input_length, output_length, x_train, y_train, x_dev, y_dev):
    if (sim_mode == "train" or sim_mode == "train_cont"):

        if (sim_mode == "train_cont"):
            # Restore the latest checkpoint in checkpoint_dir
            checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))
            print("latest checkpoint loaded")

        # get number of batches for train set
        num_batches_train = general_methods.get_num_batches(x_train.shape[0], batch_size)
        # get number of batches of dev set
        num_batches_dev = general_methods.get_num_batches(x_dev.shape[0], batch_size)

        train_start_time = time.time()
        # convert all inputs to train_step to tensors for faster processing
        # Training loop
        for epoch in range(EPOCHS):
            train_start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            for batch in range(num_batches_train):
                batch_input, batch_output = general_methods.get_batch_data(batch, batch_size, x_train, y_train)

                predicted_output, model_variables = train_step_lstm(batch_input, batch_output,
                                                                                    input_length,
                                                                                    output_length,
                                                                                    model, optimizer, train_loss,
                                                                                    train_accuracy,
                                                                                    loss_object)
                # check if model is training
                if (batch == 0):
                    old_model_variables = model_variables
                else:
                    any_change, variables_changed = general_methods.compare_lists(old_model_variables, model_variables)

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            train_time = time.time() - train_start

            # evaluation
            dev_start = time.time()
            dev_accuracy.reset_states()
            for batch in range(num_batches_dev):
                batch_input, batch_output = general_methods.get_batch_data(batch, batch_size, x_dev, y_dev)
                predicted_output = evaluate_lstm(batch_input, batch_output,
                                                                 input_length, output_length,
                                                                 model, dev_accuracy)

            dev_time = time.time() - dev_start
            print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}, \n'
                  f'dev accuracy {dev_accuracy.result():.4f}')

            print(f"for 1 epoch: train time: {train_time:.2f} secs, dev time: {dev_time:.2f} secs")
            epoch_time = train_time + dev_time
            print(f'Time taken for 1 epoch: {epoch_time:.2f} secs, in minutes: {epoch_time / 60:.2f} mins\n')

        total_train_time = time.time() - train_start_time
        # save total train time
        print("total training time = {} sec, in minutes: {} mins, in hours: {} hours".format(total_train_time,
                                                                                             total_train_time / 60,
                                                                                             total_train_time / 3600))

