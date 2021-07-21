from general_methods import load_dataset, split_sequence, train_step, forecast
from custom_layers_models import Encoder, Decoder
import tensorflow as tf
import os
import time

encoder_units = 64
decoder_units = 64
attention_units = 10
n_output_features = 1
input_length = 20
output_length = 5
batch_size = 64
EPOCHS = 4

# load the dataset
series = load_dataset()
pd.to_numeric(series)

# Initialize optimizer and loss functions
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.MeanSquaredError(reduction='none')

# create encoder & decoder instances
encoder = Encoder(encoder_units)
decoder = Decoder(decoder_units, n_output_features, attention_units)

# create a checkpoint instance
checkpoint_dir = r"C:\Users\moham\Desktop\masters\master_thesis\time_series_analysis\model_testing\cordoba checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt.txt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# create the training set
x_train, y_train = split_sequence(series, input_length, output_length)

# Restore the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
x_test = tf.expand_dims(x_train[0], axis=0)
y_test = tf.expand_dims(y_train[0], axis=0)
MAE = forecast(x_test, y_test, encoder, decoder)

x=0