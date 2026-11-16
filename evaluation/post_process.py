"""The post processing files for calculating heart rate using FFT or peak detection."""

import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags
from copy import deepcopy
import torch

def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    if len(input_signal) < 3: return input_signal
    signal_length = input_signal.shape[0]
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    try:
        detrended_signal = np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    except np.linalg.LinAlgError:
        return scipy.signal.detrend(input_signal)
    return detrended_signal

def power2db(mag):
    """Convert power to db."""
    return 10 * np.log10(mag + 1e-10)

def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.6, high_pass=3.3):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    if np.std(ppg_signal) < 1e-6: return 0.0
    
    # Zero-padding para resolução espectral
    N = _next_power_of_2(len(ppg_signal)) * 4
    
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    
    if len(mask_pxx) == 0: return 0.0
    
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    if len(ppg_peaks) < 2: return 0.0
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak

def _compute_macc(pred_signal, gt_signal):
    """(Mantido para compatibilidade de importação)"""
    if np.std(pred_signal) < 1e-6 or np.std(gt_signal) < 1e-6: return 0.0
    pred = deepcopy(pred_signal)
    gt = deepcopy(gt_signal)
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    min_len = np.min((len(pred), len(gt)))
    pred = pred[:min_len]
    gt = gt[:min_len]
    lags = np.arange(0, len(pred)-1, 1)
    tlcc_list = []
    for lag in lags:
        cross_corr = np.abs(np.corrcoef(pred, np.roll(gt, lag))[0][1])
        if np.isnan(cross_corr): cross_corr = 0
        tlcc_list.append(cross_corr)
    macc = max(tlcc_list) if len(tlcc_list) > 0 else 0
    return macc

def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.6, high_pass=3.3):
    """(Mantido para compatibilidade de importação, mas não será focado)"""
    if np.std(pred_ppg_signal) < 1e-6: return -100.0
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)
    if not signal_power_rem == 0: 
        SNR = power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR

def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """Processa o sinal e retorna o BPM."""
    
    if isinstance(predictions, torch.Tensor): predictions = predictions.detach().cpu().numpy()
    predictions = np.array(predictions)
    
    # 1. Tratamento da Predição (Onda)
    if diff_flag: 
        predictions = _detrend(np.cumsum(predictions), 100)
    else:
        predictions = _detrend(predictions, 100)
        
    if use_bandpass:
        [b, a] = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))

    # 2. Cálculo do BPM
    hr_pred = _calculate_fft_hr(predictions, fs=fs)
    
    # O Label será tratado fora daqui (no metrics.py), então retornamos 0
    hr_label = 0.0
    SNR = 0.0
    macc = 0.0
    
    return hr_label, hr_pred, SNR, macc