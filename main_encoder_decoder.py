import pandas as pd
import numpy as np
from numpy import array
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from general_methods import load_dataset, split_sequence, train_step, forecast, avg_batch_MAE
from keras.layers import Dense, LSTM, TimeDistributed, Input, RepeatVector
from keras.models import  Model
import os
import time


encoder_units = 1024
decoder_units = 1024
attention_units = 100
n_output_features = 1
n_input_features = 8
input_length = 20
output_length = 5
batch_size = 64
EPOCHS = 18

# load the dataset
series = load_dataset()
#series.apply(pd.to_numeric, errors ='ignore')

x_enc = Input(shape = (input_length, n_input_features))
#x_dec = Input(shape = (n_output_features,))
encoder = LSTM(encoder_units, activation = "relu", return_state = True)
decoder = LSTM(decoder_units, return_sequences = True, return_state = True)
output_layer = TimeDistributed(Dense(n_output_features, activation = "relu"))
repeat_vector = RepeatVector(output_length)

y_enc, h_enc, c_enc = encoder(x_enc)
x_dec = repeat_vector(y_enc)
y_dec, h_dec, c_dec = decoder(x_dec, initial_state= [h_enc, c_enc])
output_pred = output_layer(y_dec)

"""
h_dec_prev = h_enc
c_dec_prev = c_enc
x_dec = tf.expand_dims(x_dec, axis=1)
output_pred = []
for t in range(output_length):
    y_dec, h_dec, c_dec = decoder(x_dec, initial_state= [h_dec_prev, c_dec_prev])
    y_pred = output_layer(y_dec)
    output_pred.append(y_pred)

    x_dec = y_pred
    h_dec_prev = h_dec
    c_dec_prev = c_dec
"""

model = Model(x_enc, output_pred)
model.compile(optimizer="Adam", loss = "MSE", metrics = [tf.keras.metrics.RootMeanSquaredError()])
model.summary()

# create the training set
x_enc_train, y_train = split_sequence(series, input_length, output_length)
x_dec_train = tf.zeros((x_enc_train.shape[0], 1, n_output_features))

model.fit(x_enc_train, y_train, batch_size = batch_size, epochs = EPOCHS, verbose= 1, validation_split = 0.1)
x_test = tf.expand_dims(x_enc_train[0], axis=0)
y_test = tf.expand_dims(y_train[0], axis=0)
y_pred = model.predict(x_test)
y_pred = y_pred.reshape(1, -1)
fig = plt.figure()
plt.plot(y_test, label='ground truth')
plt.plot(y_pred, label='predicted output')
plt.legend()
plt.show(block=False)
q = 0

