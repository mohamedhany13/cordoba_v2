import tensorflow as tf
import logging
import numpy as np

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

class pointwise_attention(tf.keras.layers.Layer):
  def __init__(self, num_units,
               activation1='tanh', activation2 = 'relu'):
    super().__init__()
    self.Dense1 = tf.keras.layers.Dense(num_units, activation= activation1)
    self.Dense2 = tf.keras.layers.Dense(1, activation= activation2)

  def call(self, h_enc, h_dec_prev):
    # h_enc shape == (batch_size, input sequence length, hidden size of encoder)
    # h_dec_prev shape == (batch_size, hidden size of decoder)

    input_sequence_length = h_enc.shape.as_list()[1]
    # h_dec_prev with time axis shape == (batch_size, 1, hidden size of decoder)
    h_dec_prev_w_time_axis = tf.expand_dims(h_decoder_prev, axis=1)

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

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, num_units, dropout_rate=0.1):
    super().__init__()

    self.enc_lstm = tf.keras.layers.LSTM(num_units, dropout= dropout_rate, recurrent_dropout= dropout_rate,
                                         return_sequences= True, return_state= True)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)
    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_features,
               output_features, input_length, output_length, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                           input_length, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                           output_length, rate)

    self.final_layer = tf.keras.layers.Dense(output_features)

  def call(self, inp, tar, training, look_ahead_mask,
           dec_padding_mask = None, enc_padding_mask = None):

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

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
def train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy, loss_object):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp,
                                 True,
                                 combined_mask)
    loss = loss_function(tar_real, predictions, loss_object)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  AE, APE, SE = accuracy_function(tar_real, predictions)

  train_loss(loss)
  train_accuracy(APE)

  return transformer.trainable_variables
