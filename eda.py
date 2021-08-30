import numpy as np

import general_methods
import transformer_layers
import tensorflow as tf
from general_methods import load_dataset, get_resnet_units, split_sequence_autoenc, \
    avg_batch_MAE, conv_tensor_array, split_dataframe_train_dev_test, split_sequence, \
    get_input_length, plot_pred_vs_target
import matplotlib.pyplot as plt
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# type in NN architecture under investigation (this is to save different checkpoints for each architecture)
NN_arch = "transformer"
# choose OS type (linux or windows)
OS = "windows"
# choose whether to train the model or test it
# train ==> "train"
# test ==> "test"
# continue training with previous weights ==> "train_cont"
sim_mode = "train"

normalize_dataset = True

region = "Cordoba"
area_code = "06"
series = load_dataset(region, area_code, normalize= normalize_dataset, swap= False, OS = OS)

fft = tf.signal.rfft(series["Co" + area_code + "ETo"])
f_per_dataset = np.arange(0, len(fft))
#plt.step(f_per_dataset, np.abs(fft))
#plt.show(block = False)

n_samples_days = len(series["Co" + area_code + "ETo"])
days_per_year = 365.2524
years_per_dataset = n_samples_days/days_per_year

f_per_year = f_per_dataset / years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.grid()
plt.show(block = False)
plt.xscale('log')
#plt.ylim(0, 400000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 12, 365.2524], labels=['1/Year','1/month','1/day'])
_ = plt.xlabel('Frequency (log scale)')

x=0