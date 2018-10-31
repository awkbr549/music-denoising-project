#! /usr/bin/env python3

import sys
from os import listdir
from os.path import isfile, join
from math import sqrt

import soundfile
import numpy

file_base = sys.argv[1]
file_base = file_base[file_base.rfind("/")+1:file_base.rfind(".wav")]
directory = sys.argv[2]
restored_file_name_array = [f for f in listdir(directory) if(isfile(join(directory, f)) and (f != (file_base + ".wav")) and (f.startswith(file_base)) and (f.endswith(".wav")) and (not f.startswith(file_base + "_white_noise")))]
for i in range(len(restored_file_name_array)):
    restored_file_name_array[i] = directory + restored_file_name_array[i]
# for file in restored_file_name_array:
#     print(file)
# exit()

restored_file_array = []
for i in range(len(restored_file_name_array)):
    restored_file_array.append([])
    clean_data, sample_rate = soundfile.read(restored_file_name_array[i])
    restored_file_array[i].append(clean_data)
    restored_file_array[i].append(sample_rate)
    # print(restored_file_array[i])
    # print()

clean_data, sample_rate = soundfile.read(sys.argv[1])
distorted_data, sample_rate = soundfile.read(directory + file_base + "_white_noise.wav")

# @author: Sean Frankum
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

#We need an element wise measure of the prediction 
#SSE provides a decent measure of the spread of errors
def SSE(predicted, truth):
    sse = 0
    pred = predicted.ravel()
    t = truth.ravel()
    for i, p in enumerate(pred):
        sse += ((t[i] - p)**2) #math.pow((t[i] - p),2)
    #print(sse)
    return sse

#print("SNR of the clean signal: " + str(SNR(clean_data, numpy.empty((clean_data.shape[0], clean_data.shape[1])))))
#print("SNR of the distorted signal: " + str(SNR(clean_data, distorted_data - clean_data)))
#print("SNR of the distorted signal: " + str(SNR(distorted_data + numpy.full((clean_data.shape[0], clean_data.shape[1]), 1.0), distorted_data - clean_data + numpy.full((clean_data.shape[0], clean_data.shape[1]), 1.0))))
#print("SSE of the distorted signal: " + str(SSE(distorted_data, clean_data)))
#original_noise = distorted_data - clean_data
#print("Original:\t" + str(SNR(numpy.empty((clean_data.shape[0], clean_data.shape[1])), original_noise)))
#print("Original distortion:\t" + str(SNR(clean_data, distorted_data)))

print("Each restored signal: ")
for i in range(len(restored_file_name_array)):
    restored_noise = restored_file_array[i][0] - clean_data
    #rint("\tRestored Noise over Original Noise:\t" + restored_file_name_array[i][restored_file_name_array[i].rfind("/")+1:] + ":\t" + str(SNR(original_noise, restored_noise)))
    #print("\t" + restored_file_name_array[i][restored_file_name_array[i].rfind("/")+1:] + ":\t" + str(SNR(original_noise, restored_noise)))
    #print("\t" + restored_file_name_array[i][restored_file_name_array[i].rfind("/")+1:] + ":\t" + str(SNR(distorted_data - restored_file_array[i][0], clean_data - distorted_data)))
    print("\t" + restored_file_name_array[i][restored_file_name_array[i].rfind("/")+1:] + ":\t" + str(SNR(distorted_data - restored_file_array[i][0], numpy.ones((clean_data.shape[0], clean_data.shape[1])))**0.5))

    #print("\tSNR:\t" + restored_file_name_array[i][restored_file_name_array[i].rfind("/")+1:] + ":\t" + str(SNR(clean_data, restored_file_array[i][0] - clean_data)))
    #print("\tSNR:\t" + restored_file_name_array[i][restored_file_name_array[i].rfind("/")+1:] + ":\t" + str(SNR(clean_data + numpy.full((clean_data.shape[0], clean_data.shape[1]), 1.0), restored_file_array[i][0] - clean_data + numpy.full((clean_data.shape[0], clean_data.shape[1]), 1.0))))
    #print("\tSSE:\t" + restored_file_name_array[i][restored_file_name_array[i].rfind("/")+1:] + ":\t" + str(SSE(restored_file_array[i][0], clean_data)))
    #print()