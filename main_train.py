from general_methods import load_dataset, split_sequence, train_step, forecast, avg_batch_MAE
from custom_layers_models import Encoder, Decoder
import tensorflow as tf
import os
import time

encoder_units = 128
decoder_units = 128
attention_units = 20
n_output_features = 1
input_length = 20
output_length = 5
batch_size = 64
EPOCHS = 20

# load the dataset
series = load_dataset()

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

# get number of batches
num_batches = int(x_train.shape[0] / batch_size)
remainder = x_train.shape[0] % batch_size
if (remainder != 0):
    num_batches += 1

# Training loop
for epoch in range(EPOCHS):
    start = time.time()

    # Initialize the hidden state
    # enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    # Loop through the dataset
    for batch in range(num_batches):

        # get the current batch of x_train & y_train
        start_index = batch * batch_size
        end_index = start_index + batch_size
        if (end_index > x_train.shape[0]):
            end_index = x_train.shape[0]
        batch_input = x_train[start_index: end_index]
        batch_output = y_train[start_index: end_index]
        #batch_output = tf.expand_dims(batch_output, axis=1)

        # Call the train method
        output_pred, batch_loss, trainable_variables = train_step(batch_input, batch_output, encoder, decoder, optimizer, loss_object)

        # Compute the loss (per epoch)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

    # Save (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    # Output the loss observed until that epoch
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / num_batches))

    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

# Restore the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
x_test = tf.expand_dims(x_train[0], axis=0)
y_test = tf.expand_dims(y_train[0], axis=0)
MAE = forecast(x_test, y_test, encoder, decoder)

x=0