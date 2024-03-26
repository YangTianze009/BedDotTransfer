#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:55:05 2024

@author: Yingjian
"""

import numpy as np
from DSP_func import signal_quality_assessment, get_envelope

# data_type = "test"
# data_path = f"./day_data_{data_type}.npy"

data_path = "../../datasets/stable_noise02/simu_10k.npy"

#### load dataset
dataset = np.load(data_path)

### obtain signal and discard label
data = dataset[:, :1000]
data_labels = dataset[:, 1000:]

##### pre-define band pass filter cutoff frequency
low = 1
high = 15

##### set sampling rate
Fs = 100

good_data = []
extracted_envelope = []
good_data_labels = []
for i in range(data.shape[0]):
    temp_x = data[i]
    temp_y = data_labels[i]
    ##### assess signal quality and filter out bad quality data
    res = signal_quality_assessment(
        x=temp_x, Fs=Fs, n_lag=len(temp_x) // 2, low=low, high=high
    )
    if res[0]:  ### good data condition
        good_data.append(temp_x)
        good_data_labels.append(temp_y)
        temp_envelope = get_envelope(x=temp_x, Fs=Fs, low=low, high=high)
        extracted_envelope.append(temp_envelope)

####obtain all data with good quality
good_data = np.array(good_data)
#### extracted envelope from all good quality data
extracted_envelope = np.array(extracted_envelope)
print(extracted_envelope.shape)
final_data = np.column_stack((extracted_envelope, good_data_labels))
print(final_data.shape)
# np.save(f"extracted_envelope_data_{data_type}.npy", final_data)
np.save(
    f"../../datasets/stable_noise02/envelope_data/extracted_envelope_simu_10k.npy",
    final_data,
)
