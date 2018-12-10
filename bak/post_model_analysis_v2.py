#! /usr/bin/env python3

import sys
from os import listdir
from os.path import isfile, join
import numpy
import soundfile
from math import sqrt
from operator import itemgetter

def SNR(signal, noise):
    #SNR is the ratio of signal and noise strength
    num_samples = signal.shape[0] * signal.shape[1]
    if (signal.shape != noise.shape):
        print('error, signal and noise should be same size')
        return
    
    r_signal = signal.ravel()
    r_noise = noise.ravel()

    #calculating RMS of signal
    s_rms = 0.0
    for s in r_signal:
        s_rms += (s * s)
    s_rms = sqrt(s_rms / num_samples)
    #print("\t\ts_rms: " + str(s_rms))

    #calculating the RMS of noise
    n_rms = 0.0
    for n in r_noise:
        n_rms += (n * n)
    n_rms = sqrt(n_rms / num_samples)
    #print("\t\tn_rms: " + str(n_rms))

    result = numpy.float64("inf")
    if (n_rms > 0.0):
        result = ((s_rms / n_rms) ** 2)
    return result

def SSE(predicted, truth):
    sse = 0
    pred = predicted.ravel()
    t = truth.ravel()
    for i in range(len(pred)):
        sse += ((pred[i] - t[i])**2)

    # pred = predicted.ravel()
    # t = truth.ravel()
    # for i, p in enumerate(pred):
    #     sse += math.pow((t[i] - p),2)
    # print(sse)

    return sse

clean_file_name = sys.argv[1]
clean_data, sample_rate = soundfile.read(clean_file_name)
clean_file_name = clean_file_name[clean_file_name.rfind("/")+1:clean_file_name.rfind(".wav")]
directory = sys.argv[2]

restored_file_name_array = [f for f in listdir(directory) if (isfile(join(directory, f)) and (f.endswith(".wav")))]
for i in range(len(restored_file_name_array)):
    restored_file_name_array[i] = [join(directory, restored_file_name_array[i]), 0.0, 0.0] #filename, SNR, SSE

for filename in restored_file_name_array:
    temp_restored = soundfile.read(filename[0])[0]
    filename[1] = round(SNR(clean_data, clean_data - temp_restored), 4)
    filename[2] = round(SSE(temp_restored, clean_data), 4)

# snr_max_mov_avg = 0
# snr_min_mov_avg = 0
# snr_max_rf = 0
# snr_min_rf = 0
# sse_max_mov_avg = 0
# sse_min_mov_avg = 0
# sse_max_rf = 0
# sse_min_rf = 0

# for i in range(1, len(restored_file_name_array)):
#     if (restored_file_name_array[i][0].find("mov_avg") >= 0):
#         #moving average
#         if (restored_file_name_array[i][1] > restored_file_name_array[snr_max_mov_avg][1]):
#             snr_max_mov_avg = i
#         if (restored_file_name_array[i][1] < restored_file_name_array[snr_min_mov_avg][1]):
#             snr_min_mov_avg = i
#         if (restored_file_name_array[i][2] > restored_file_name_array[sse_max_mov_avg][2]):
#             sse_max_mov_avg = i
#         if (restored_file_name_array[i][2] < restored_file_name_array[sse_min_mov_avg][2]):
#             sse_min_mov_avg = i
#     elif (restored_file_name_array[i][0].find("md_") >= 0):
#         #random forest
#         if (restored_file_name_array[i][1] > restored_file_name_array[snr_max_rf][1]):
#             snr_max_rf = i
#         if (restored_file_name_array[i][1] < restored_file_name_array[snr_min_rf][1]):
#             snr_min_rf = i
#         if (restored_file_name_array[i][2] > restored_file_name_array[sse_max_rf][2]):
#             sse_max_rf = i
#         if (restored_file_name_array[i][2] < restored_file_name_array[sse_min_rf][2]):
#             sse_min_rf = i

# print("mov_avg max SNR: " + str(restored_file_name_array[snr_max_mov_avg]))
# print("mov_avg min SNR: " + str(restored_file_name_array[snr_min_mov_avg]))
# print("mov_avg max SSE: " + str(restored_file_name_array[sse_max_mov_avg]))
# print("mov_avg min SSE: " + str(restored_file_name_array[sse_min_mov_avg]))
# print("rf max SNR: " + str(restored_file_name_array[snr_max_rf]))
# print("rf min SNR: " + str(restored_file_name_array[snr_min_rf]))
# print("rf max SSE: " + str(restored_file_name_array[sse_max_rf]))
# print("rf min SSE: " + str(restored_file_name_array[sse_min_rf]))


snr_sorted = sorted(restored_file_name_array, key=itemgetter(1))
sse_sorted = sorted(restored_file_name_array, key=itemgetter(2))
for filename in snr_sorted:
    print(filename)
print()
for filename in sse_sorted:
    print(filename)
# print("snr_sorted[0]: " + str(snr_sorted[0]))
# print("snr_sorted[1]: " + str(snr_sorted[len(snr_sorted)-1]))
# print()
# print("sse_sorted[0]: " + str(sse_sorted[0]))
# print("sse_sorted[1]: " + str(sse_sorted[len(sse_sorted)-1]))
