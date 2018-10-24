#! /usr/bin/env python3

import sys
from playsound import playsound
import soundfile
import numpy
#from sklearn.feature_extraction.image import extract_patches_2d

from sklearn.ensemble import RandomForestRegressor

input_file_name = sys.argv[1]
output_file_name_white = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_white_noise.wav"

print("Extracting data... ", end="")
sys.stdout.flush()
clean_data, sample_rate = soundfile.read(input_file_name) #returns float64 encoding
print("Done.")

# print("Original file playing... ", end="")
# sys.stdout.flush()
# playsound(input_file_name)
# print("Done.")

print("Noising data with white noise... ", end="")
sys.stdout.flush()
distorted = clean_data.copy()
height, width = distorted.shape
white_noise = 0.005 * numpy.random.randn(height, width) #returns matrix with 0 mean and 1 variance
soundfile.write("output/white_noise.wav", white_noise, sample_rate)
distorted += white_noise
soundfile.write(output_file_name_white, distorted, sample_rate)
print("Done.")

# print("Playing file with white noise added... ", end="")
# sys.stdout.flush()
# playsound(output_file_name_white)
# print("Done.")

print("Training forest... ", end="")
sys.stdout.flush()
X = numpy.empty((0, 5))
y = numpy.empty(0)
for t in range(2000, 2100):
    #X = numpy.concatenate((X, numpy.asarray([[clean_data[t][0], clean_data[t+1][0], clean_data[t+2][0], clean_data[t+3][0], clean_data[t+4][0]]])))
    X = numpy.concatenate((X, numpy.asarray([[distorted[t][0], distorted[t+1][0], distorted[t+2][0], distorted[t+3][0], distorted[t+4][0]]])))
    #X = numpy.concatenate((X, numpy.asarray([[distorted[t][1], distorted[t+1][1], distorted[t+2][1], distorted[t+3][1], distorted[t+4][1]]])))
    y = numpy.concatenate((y, numpy.asarray([clean_data[t+5][0]])))
    #y = numpy.concatenate((y, numpy.asarray([distorted[t+5][1]])))
    #y = numpy.concatenate((y, numpy.asarray([distorted[t+5][0]])))

regr = RandomForestRegressor(max_depth=7, random_state=1, n_estimators=100)
regr.fit(X, y)
# print(regr.feature_importances_)
print("Done.")

# print("Attempting to remove white noise from data... ", end="")
# sys.stdout.flush()
restored = numpy.empty((height, width))

temp_avg = 0
temp_sse = 0
# X = clean, y = clean, predict(clean)
# X = clean, y = clean, predict(distorted)
# X = distorted, y = clean, predict(distorted)
# X = distorted, y = distorted, predict(distorted)
for t in range(2000, 2100):    
    #diff = clean_data[t+5][0] - regr.predict([[clean_data[t][0], clean_data[t+1][0], clean_data[t+2][0], clean_data[t+3][0], clean_data[t+4][0]]])[0]
    diff = clean_data[t+5][0] - regr.predict([[distorted[t][0], distorted[t+1][0], distorted[t+2][0], distorted[t+3][0], distorted[t+4][0]]])[0]
    #print("Difference: " + str(diff))
    temp_avg += diff
    temp_sse += (diff**2)

print()
temp_avg /= 100
print("Average difference: " + str(temp_avg))
print("SSE: " + str(temp_sse))