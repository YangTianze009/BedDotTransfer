#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:55:05 2024

@author: Yingjian, Vatsal Thakkar <vatsalthakkar3.vt@gmail.com>
"""
import os
import numpy as np
from sympy import Float
from DSP_func import signal_quality_assessment, get_envelope


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory {path} created")
    else:
        print(f"Directory {path} already exists")


def main():

    noise_levels = ["00", "02", "04", "06", "08", "10"]
    window_size = [
        "0.040",
        "0.080",
        "0.160",
        "0.200",
        "0.250",
        "0.280",
        "0.300",
        "0.350",
    ]
    data_split = ["10", "5", "2"]  # Train : 10k , Test: 5k, val : 2k
    for noise in noise_levels:
        for split in data_split:
            data_path = f"datasets/stable_noise{noise}/simu_{split}k.npy"
            for window in window_size:
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
                        temp_envelope = get_envelope(
                            x=temp_x, Fs=Fs, low=low, high=high, window=Float(window)
                        )
                        extracted_envelope.append(temp_envelope)

                ####obtain all data with good quality
                good_data = np.array(good_data)
                #### extracted envelope from all good quality data
                extracted_envelope = np.array(extracted_envelope)
                print(extracted_envelope.shape)
                final_data = np.column_stack((extracted_envelope, good_data_labels))
                print(final_data.shape)
                create_directory(
                    f"datasets/stable_noise{noise}/envelope_data/{window.replace('.','_')}"
                )
                np.save(
                    f"datasets/stable_noise{noise}/envelope_data/{window.replace('.','_')}/extracted_envelope_data_{split}k.npy",
                    final_data,
                )
                print(
                    "File saved at : ",
                    f"datasets/stable_noise{noise}/envelope_data/{window.replace('.','_')}",
                )


if __name__ == "__main__":
    main()
