import numpy as np

import general_methods
import lstm_att_layers
import tensorflow as tf
import os
import time
import pandas as pd
import logging
import graphviz
import pydot
from matplotlib import pyplot as plt
import statsmodels
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# type in NN architecture under investigation (this is to save different checkpoints for each architecture)
NN_arch = "lstm"
# choose OS type (linux or windows)
OS = "windows"
# choose 1 to use path in drive, 0 to use path in downloads (for linux only)
linux_path = 0
# choose whether to train the model or test it
# train ==> "train"
# test ==> "test"
# continue training with previous weights ==> "train_cont"
sim_mode = "train"

# choose zero if you want to set it to default
learning_rate = 0.001
block_shuffle_size = 10

normalize_dataset = False
input_length = 60
output_length = 1
input_features = 8
output_features = 1
validation_split = 0.2
test_split = 0.1
batch_size = 128
EPOCHS = 500

lstm_units = 256
attention_units = 128
dropout_rate = 0.1

region = "Cordoba"
area_code = "06"
series = general_methods.load_dataset(region, area_code, normalize= normalize_dataset, swap= False, OS = OS,
                                      linux_path = linux_path)

normal_series = series["Co" + area_code + "ETo"]
import numpy as np

import general_methods
import lstm_att_layers
import tensorflow as tf
import os
import time
import pandas as pd
import logging
import graphviz
import pydot
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# type in NN architecture under investigation (this is to save different checkpoints for each architecture)
NN_arch = "lstm"
# choose OS type (linux or windows)
OS = "windows"
# choose 1 to use path in drive, 0 to use path in downloads (for linux only)
linux_path = 0
# choose whether to train the model or test it
# train ==> "train"
# test ==> "test"
# continue training with previous weights ==> "train_cont"
sim_mode = "train"

# choose zero if you want to set it to default
learning_rate = 0.001
block_shuffle_size = 10

normalize_dataset = False
input_length = 60
output_length = 1
input_features = 8
output_features = 1
validation_split = 0.2
test_split = 0.1
batch_size = 128
EPOCHS = 500

lstm_units = 256
attention_units = 128
dropout_rate = 0.1

region = "Cordoba"
area_code = "06"

def plot_real_vs_predicted(real, predicted, title):
    plt.figure()
    x_plot = np.arange(len(real))
    plt.plot(x_plot, real, 'r', label="real")
    plt.plot(x_plot, predicted, 'b', label="predicted")
    plt.xlabel('time')
    plt.ylabel('evapotranspiration')
    plt.title(title)
    plt.legend()
    plt.show(block=False)

def get_MAPE(real, predicted, num_predictions):
    MAPE = 0
    for i in range(len(real) - num_predictions, len(real)):
        MAPE += (np.absolute(predicted[i] - real[i]) / real[i]) * 100

    MAPE /= num_predictions
    return MAPE

series = general_methods.load_dataset(region, area_code, normalize= normalize_dataset, swap= False, OS = OS,
                                      linux_path = linux_path)

evapo_series = series["Co" + area_code + "ETo"].values
Snaive_predicted = np.zeros((len(evapo_series)))

# predict 2 years in future (365 * 2 = 730)
Snaive_predicted[: -730] = evapo_series[: -730]
for i in range(len(evapo_series) - 730, len(evapo_series)):
    Snaive_predicted[i] = evapo_series[i - 365]

Snaive_MAPE = get_MAPE(evapo_series, Snaive_predicted, 730)
plot_real_vs_predicted(evapo_series, Snaive_predicted, "Snaive prediction")


# getting trend
plt.figure()
from statsmodels.tsa.seasonal import seasonal_decompose
evapo_series_no_zeros = series["Co" + area_code + "ETo"][series["Co" + area_code + "ETo"].values != 0]
evapo_trend = evapo_series_no_zeros.rolling(window=365).mean()
evapo_trend = evapo_trend.rolling(window=2).mean()
evapo_trend.plot()
plt.show(block = False)

# simple exponential smoothing (SES)
def SES(input, current_predicted_series_size, alpha):
    SES_coefficients = np.zeros((current_predicted_series_size))
    for i in range(current_predicted_series_size - 1, 0 - 1, -1):
        exponent = current_predicted_series_size - 1 - i
        SES_coefficients[i] = alpha * pow((1 - alpha), exponent)
    mult_output = np.multiply(input, SES_coefficients)
    predicted_output = sum(mult_output)
    return predicted_output, SES_coefficients


SES_predicted = np.zeros((len(evapo_series)))
alpha = 0.998
SES_predicted[: -730] = evapo_series[: -730]

for i in range(len(evapo_series) - 730, len(evapo_series)):
    SES_predicted[i], SES_coefficients = SES(SES_predicted[:i], i, alpha)
SES_MAPE = get_MAPE(evapo_series, SES_predicted, 730)
plot_real_vs_predicted(evapo_series, SES_predicted, "SES prediction")

# seasonal decomposition
result= statsmodels.tsa.seasonal.seasonal_decompose(evapo_series, model='additive', period=365)
residuals = evapo_series - result.seasonal
from statsmodels.graphics import tsaplots
plt.figure()
pd.plotting.autocorrelation_plot(residuals)
plt.show(block = False)
tsaplots.plot_pacf(residuals, lags = 1095)
plt.show(block = False)


# plot pacf of evapotranspiration
from statsmodels.graphics import tsaplots
plt.figure()
tsaplots.plot_acf(evapo_series_no_zeros.values, lags = 5000)
plt.show(block = False)
plt.figure()
pd.plotting.autocorrelation_plot(evapo_series_no_zeros)
plt.show(block = False)
pacf_start = time.time()
tsaplots.plot_pacf(evapo_series_no_zeros.values, lags = 1095)
plt.show(block = False)
pacf_time = time.time() - pacf_start

# acf & pacf of seasonally differenced data
seasonaly_diff_series = np.zeros((len(evapo_series_no_zeros) - 365))
for i in range(len(evapo_series_no_zeros) - 365):
    seasonaly_diff_series[i] = evapo_series_no_zeros[i + 365] - evapo_series_no_zeros[i]
plt.figure()
plt.plot(np.arange(len(seasonaly_diff_series)), seasonaly_diff_series, 'b', label="real")
plt.show(block = False)
tsaplots.plot_acf(seasonaly_diff_series, lags = 5000)
plt.show(block = False)
tsaplots.plot_pacf(seasonaly_diff_series, lags = 1095)
plt.show(block = False)

# acf & pacf of seasonally differenced data with first difference
seasonaly_diff_series = np.zeros((len(evapo_series_no_zeros) - 365))
for i in range(len(evapo_series_no_zeros) - 365):
    seasonaly_diff_series[i] = evapo_series_no_zeros[i + 365] - evapo_series_no_zeros[i]
sd_series_v1 = np.zeros((len(seasonaly_diff_series) - 1))
for i in range(len(sd_series_v1)):
    sd_series_v1[i] = seasonaly_diff_series[i + 1] - seasonaly_diff_series[i]
plt.figure()
plt.plot(np.arange(len(sd_series_v1)), sd_series_v1, 'b', label="blue")
plt.show(block = False)
tsaplots.plot_acf(sd_series_v1, lags = 5000)
plt.show(block = False)
tsaplots.plot_pacf(sd_series_v1, lags = 1095)
plt.show(block = False)





x = 0