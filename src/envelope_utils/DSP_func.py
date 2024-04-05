#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:45:16 2024

@author: Yingjian
"""
import numpy as np
import pywt
from scipy import signal
from scipy.signal import hilbert, savgol_filter, periodogram
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt


def moving_window_integration(signal, fs):

    # Initialize result and window size for integration
    result = signal.copy()
    win_size = round(0.20 * fs)
    sum = 0

    # Calculate the sum for the first N terms
    for j in range(win_size):
        sum += signal[j] / win_size
        result[j] = sum

    # Apply the moving window integration using the equation given
    for index in range(win_size, len(signal)):
        sum += signal[index] / win_size
        sum -= signal[index - win_size] / win_size
        result[index] = sum

    return result


def get_envelope(x, Fs, low, high, m_wave="db12", denoised_method="bandpass"):
    x = (x - np.mean(x)) / np.std(x)
    if denoised_method == "DWT":
        denoised_sig = wavelet_reconstruction(x=x, fs=Fs, low=low, high=high)

    elif denoised_method == "bandpass":
        denoised_sig = band_pass_filter(data=x, Fs=100, low=low, high=high, order=3)

    z = hilbert(denoised_sig)  # form the analytical signal
    envelope = np.abs(z)

    smoothed_envelope = moving_window_integration(signal=envelope, fs=Fs)
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope)) / np.std(
        smoothed_envelope
    )
    # # smoothed_envelope = detrend(smoothed_envelope)
    # smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope))/np.std(smoothed_envelope)

    return smoothed_envelope


def wavelet_decomposition(data, wave, Fs=None, n_decomposition=None):
    a = data
    w = wave
    ca = []
    cd = []
    rec_a = []
    rec_d = []
    freq_range = []
    for i in range(n_decomposition):
        if i == 0:
            freq_range.append(Fs / 2)
        freq_range.append(Fs / 2 / (2 ** (i + 1)))
        (a, d) = pywt.dwt(a, w)
        ca.append(a)
        cd.append(d)

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec = pywt.waverec(coeff_list, w)
        rec_a.append(rec)
        # ax3[i].plot(Fre, FFT_y1)
        # print(max_freq)

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    return rec_a, rec_d


def freq_com_select(Fs, low, high):
    n = 0
    valid_freq = Fs / 2
    temp_f = valid_freq
    min_diff_high = abs(temp_f - high)
    min_diff_low = abs(temp_f - low)

    while temp_f > low:
        temp_f = temp_f / 2
        n += 1
        diff_high = abs(temp_f - high)
        diff_low = abs(temp_f - low)
        if diff_high < min_diff_high:
            max_n = n
            min_diff_high = diff_high
        if diff_low < min_diff_low:
            min_n = n
            min_diff_low = diff_low
    return n, max_n, min_n


def wavelet_reconstruction(x, fs, low, high):
    n_decomposition, max_n, min_n = freq_com_select(Fs=fs, low=low, high=high)
    rec_a, rec_d = wavelet_decomposition(
        data=x, wave="db12", Fs=fs, n_decomposition=n_decomposition
    )
    min_len = len(rec_d[max_n])
    for n in range(max_n, min_n):
        if n == max_n:
            denoised_sig = rec_d[n][:min_len]
        else:
            denoised_sig += rec_d[n][:min_len]
    cut_len = (len(denoised_sig) - len(x)) // 2
    denoised_sig = denoised_sig[cut_len:-cut_len]
    return denoised_sig


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def band_pass_filter(data, Fs, low, high, order):
    b, a = signal.butter(order, [low / (Fs * 0.5), high / (Fs * 0.5)], "bandpass")
    # perform band pass filter
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def signal_quality_assessment(
    x, Fs, n_lag, low, high, denoised_method="bandpass", show=False
):
    x = (x - np.mean(x)) / np.std(x)
    if denoised_method == "DWT":
        denoised_sig = wavelet_reconstruction(x=x, fs=Fs, low=low, high=high)

    elif denoised_method == "bandpass":
        denoised_sig = band_pass_filter(data=x, Fs=100, low=0.6, high=10, order=5)
    index = 0
    window_size = int(Fs)
    z = hilbert(denoised_sig)  # form the analytical signal
    envelope = np.abs(z)

    sg_win_len = round(0.41 * Fs)
    if sg_win_len % 2 == 0:
        sg_win_len -= 1
    smoothed_envelope = savgol_filter(envelope, sg_win_len, 3, mode="nearest")
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope)) / np.std(
        smoothed_envelope
    )

    sg_win_len = round(2.01 * Fs)
    if sg_win_len % 2 == 0:
        sg_win_len -= 1
    trend = savgol_filter(smoothed_envelope, sg_win_len, 3, mode="nearest")
    smoothed_envelope = smoothed_envelope - trend
    smoothed_envelope = (smoothed_envelope - np.mean(smoothed_envelope)) / np.std(
        smoothed_envelope
    )
    acf_x = acf(smoothed_envelope, nlags=n_lag)
    acf_x = acf_x / acf_x[0]

    nfft = next_power_of_2(x=len(x) * 2)
    f, Pxx_den = periodogram(acf_x, fs=Fs, nfft=nfft)

    sig_means = []
    index = 0
    frequency = f[np.argmax(Pxx_den)]
    power = max(Pxx_den)
    if show:
        fig, ax = plt.subplots(6, 1, figsize=(16, 18))
        ax[0].plot(x, label="raw data")
        ax[1].plot(denoised_sig, label="wavelet denoised data")
        ax[2].plot(envelope, label="envelope extraction by Hilbert transform")
        ax[3].plot(smoothed_envelope, label="smoothed envelope")
        ax[4].plot(acf_x, label="ACF of smoothed envelope")
        ax[5].plot(f, Pxx_den, label="spectrum of ACF")
        for i in range(len(ax)):
            ax[i].legend()

    while index + window_size < len(acf_x):
        sig_means.append(np.mean(acf_x[index : index + window_size]))
        index = index + window_size
    if np.std(sig_means) < 0.1 and 0.6 < frequency < 3 and power > 0.1:
        res = [True, np.std(sig_means), frequency, power, acf_x]
        if show:
            fig.suptitle("good data")
    else:
        res = [False, np.std(sig_means), frequency, power, acf_x]
        if show:
            fig.suptitle("bad data")
    return res
