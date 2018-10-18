#! /usr/bin/env python3

import sys
from playsound import playsound
import soundfile
import numpy
from sklearn.feature_extraction.image import extract_patches_2d

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

X = numpy.empty((0, 1))
y = numpy.empty(0)
for t in range(1000, 1100):
    X = numpy.concatenate((X, numpy.asarray([[clean_data[t][0]]])))
    y = numpy.concatenate((y, numpy.asarray([clean_data[t+1][0]])))

# patch_size = (100, 2)
# patches = extract_patches_2d(clean_data, patch_size)
# print(patches.shape)
# #print(patches[0].shape)

# X = numpy.empty((0, 1))
# y = numpy.empty(0)
# for t in range(1000, 1000 + patch_size[0]):
#     X = numpy.concatenate((X, numpy.asarray([[t]])))
#     y = numpy.concatenate((y, numpy.asarray([patches[t][0][0]])))

regr = RandomForestRegressor(max_depth=8, random_state=2, n_estimators=100).fit(X, y)

for t in range(1000, 1100):
    print()
    print(clean_data[t+1][0])
    print(y[t - 1000])
    print(regr.predict([[clean_data[t][0]]])[0])