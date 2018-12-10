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
distorted_file_name = sys.argv[2]
distorted, sample_rate = soundfile.read(distorted_file_name)
original_noise = clean_data - distorted
print("SNR of distorted: " + str(round(SNR(clean_data, original_noise), 4)))
print("SSE of distorted: " + str(round(SSE(distorted, clean_data), 4)))

clean_file_name = clean_file_name[clean_file_name.rfind("/")+1:clean_file_name.rfind(".wav")]
directory = "output" + distorted_file_name[distorted_file_name.rfind("/"):len(distorted_file_name)-4] + "/"
#print(directory)
#exit()

restored_file_name_array = [f for f in listdir(directory) if (isfile(join(directory, f)) and (f.endswith(".wav")) and
    (not f.find("rs_1") > -1 or not f.find("rs_2")) 
    #((f.find("ne_100") > -1) or (f.find("ne_200") > -1) or (f.find("mov") > -1)) and
    #((f.find("md_8") > -1) or (f.find("md_16") > -1) or (f.find("mov") > -1))
    )]
for i in range(len(restored_file_name_array)):
    restored_file_name_array[i] = [join(directory, restored_file_name_array[i]), 0.0, 0.0] #filename, SNR, SSE

for filename in restored_file_name_array:
    temp_restored = soundfile.read(filename[0])[0]
    noise_reduction = temp_restored - distorted
    post_noise = original_noise - noise_reduction
    filename[1] = round(SNR(clean_data, post_noise), 4)
    filename[2] = round(SSE(temp_restored, clean_data), 4)

#sorting by SNR or SSE and printing
# snr_sorted = sorted(restored_file_name_array, key=itemgetter(1))
# sse_sorted = sorted(restored_file_name_array, key=itemgetter(2))
# print("SNR sorted: SNR, filename")
# for filename in snr_sorted:
#     print(str(filename[1]) + ", " + filename[0])
# print("\nSSE sorted: SSE, filename")
# for filename in sse_sorted:
#     print(str(filename[2]) + ", " + filename[0])

print("filename, SNR, SSE")
alpha_sorted = sorted(restored_file_name_array, key=itemgetter(0))
for filename in alpha_sorted:
    print(filename)
