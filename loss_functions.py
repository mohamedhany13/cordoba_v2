import tensorflow as tf

# Loss function
def loss_function(real, pred, loss_object):

  # If there's a '0' in the sequence, the loss is being nullified
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  loss_reshaped = tf.expand_dims(loss_, axis = 1)

  mask = tf.cast(mask, dtype=loss_.dtype)
  #loss_ *= mask
  loss_wo_zeros = tf.math.multiply(loss_reshaped, mask)
  mean_loss = tf.math.reduce_mean(loss_wo_zeros)

  return mean_loss
# mean batch loss?