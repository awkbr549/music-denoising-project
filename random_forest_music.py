#! /usr/bin/env python3

import sys
from playsound import playsound
import soundfile
import numpy
#from sklearn.feature_extraction.image import extract_patches_2d

from sklearn.ensemble import RandomForestRegressor

input_file_name = sys.argv[1]
window_size = int(sys.argv[2])
#if window_size isn't even, round up
if (window_size % 2 == 1):
    window_size += 1

output_file_name_white = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_white_noise.wav"
output_file_name_random_forest_left = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_random_forest_left_" + str(window_size) + ".wav"
output_file_name_random_forest_center = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_random_forest_center_" + str(window_size) + ".wav"
output_file_name_random_forest_right = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_random_forest_right" + str(window_size) + ".wav"


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


print("Training forests... ", end="")
sys.stdout.flush()
restored_array = [
    numpy.empty((height, width)), #clean input, clean output, left window
    numpy.empty((height, width)), #clean input, clean output, center window
    numpy.empty((height, width)), #clean input, clean output, right window
    numpy.empty((height, width)), #distorted input, clean output, left window
    numpy.empty((height, width)), #distorted input, clean output, center window
    numpy.empty((height, width)), #distorted input, clean output, right window
    numpy.empty((height, width)), #distorted input, distorted output, left window
    numpy.empty((height, width)), #distorted input, distorted output, center window
    numpy.empty((height, width)) #distorted input, distorted output, right window
    ]

restored_left = numpy.empty((height, width))
restored_center = numpy.empty((height, width))
restored_right = numpy.empty((height, width))
MAX_DEPTH = 7
RANDOM_STATE = 1
N_ESTIMATORS = 100
random_forest_array = [
    RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #clean input, clean output, left window
    RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #clean input, clean output, center window
    RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #clean input, clean output, right window
    RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #distorted input, clean output, left window
    RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #distorted input, clean output, center window
    RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #distorted input, clean output, right window
    RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #distorted input, distorted output, left window
    RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #distorted input, distorted output, center window
    RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS) #distorted input, distorted output, right window
    ]

for c in range(width):
    X_array = [
        numpy.empty((0, 5)), #clean input, left window
        numpy.empty((0, 5)), #clean input, center window
        numpy.empty((0, 5)), #clean input, right window
        numpy.empty((0, 5)), #distorted input, left window
        numpy.empty((0, 5)), #distorted input, center window
        numpy.empty((0, 5)) #distorted input, right window
        ]
    y_array = [
        numpy.empty(0), #clean output, left window
        numpy.empty(0), #clean output, center window
        numpy.empty(0), #clean output, right window
        numpy.empty(0), #distorted output, left window
        numpy.empty(0), #distorted output, center window
        numpy.empty(0) #distorted outtput, right window
        ]

    X_list_array = [
        [], #clean input, left window
        [], #clean input, center window
        [], #clean input, right window
        [], #distorted input, left window
        [], #distorted input, center window
        [] #distorted input, right window
        ]
    y_list_array = [
        [], #clean output, left window
        [], #clean output, center window
        [], #clean output, right window
        [], #distorted output, left window
        [], #distorted output, center window
        [] #distorted output, right window
        ]

    for t in range(height - window_size):
        for i in range(window_size):
            #training on clean input
            X_list_array[0].append(clean[t+i][c]) #clean input, left window
            if (i < window_size / 2):
                X_list_array[1].append(clean[t+i][c]) #clean input, center window
            else:
                X_list_array[1].append(clean[t+i+1][c]) #clean input, center window
            X_list_array[2].append(clean[t+i+1][c]) #clean input, right window

            #training on distorted input
            X_list_array[3].append(distorted[t+i][c]) #distorted input, left window
            if (i < window_size / 2):
                X_list_array[4].append(clean[t+i][c]) #distorted input, center window
            else:
                X_list_array[4].append(clean[t+i+1][c]) #distorted input, center window
            X_list_array[5].append(clean[t+i+1][c]) #distorted input, right window

        #training on clean output
        y_list_array[0].append((clean[t+window_size][c])) #clean output, left window
        y_list_array[1].append((clean[t+int(window_size/2)][c])) #clean output, center window
        y_list_array[2].append((clean[t][c])) #clean output, right window

        #training on distorted output
        y_list_array[3].append((distorted[t+window_size][c])) #distorted output, left window
        y_list_array[4].append((distorted[t+int(window_size/2)][c])) #distorted output, center window
        y_list_array[5].append((distorted[t][c])) #distorted output, right window


    X_array[0] = numpy.concatenate((X_array[0], numpy.asarray([X_list_array[0]]))) #clean input, left window
    X_array[1] = numpy.concatenate((X_array[1], numpy.asarray([X_list_array[1]]))) #clean input, center window
    X_array[2] = numpy.concatenate((X_array[2], numpy.asarray([X_list_array[2]]))) #clean input, right window
    X_array[3] = numpy.concatenate((X_array[3], numpy.asarray([X_list_array[3]]))) #distorted input, left window
    X_array[4] = numpy.concatenate((X_array[4], numpy.asarray([X_list_array[4]]))) #distorted input, center window
    X_array[5] = numpy.concatenate((X_array[5], numpy.asarray([X_list_array[5]]))) #distorted input, right window

    y_array[0] = numpy.concatenate((y_array[0], numpy.asarray(y_list_array[0]))) #clean output, left window
    y_array[1] = numpy.concatenate((y_array[1], numpy.asarray(y_list_array[1]))) #clean output, center window
    y_array[2] = numpy.concatenate((y_array[2], numpy.asarray(y_list_array[2]))) #clean output, right window
    y_array[3] = numpy.concatenate((y_array[3], numpy.asarray(y_list_array[3]))) #distorted output, left window
    y_array[4] = numpy.concatenate((y_array[4], numpy.asarray(y_list_array[4]))) #distorted output, center window
    y_array[5] = numpy.concatenate((y_array[5], numpy.asarray(y_list_array[5]))) #distorted output, right window

    random_forest_array[0].fit(X_array[0], y_array[0]) #clean input, clean output, left window
    random_forest_array[1].fit(X_array[1], y_array[1]) #clean input, clean output, center window
    random_forest_array[2].fit(X_array[2], y_array[2]) #clean input, clean output, right window
    random_forest_array[3].fit(X_array[3], y_array[0]) #distorted input, clean output, left window
    random_forest_array[4].fit(X_array[4], y_array[1]) #distorted input, clean output, center window
    random_forest_array[5].fit(X_array[5], y_array[2]) #distorted input, clean output, right window
    random_forest_array[6].fit(X_array[3], y_array[3]) #distorted input, distorted output, left window
    random_forest_array[7].fit(X_array[4], y_array[4]) #distorted input, distorted output, center window,
    random_forest_array[8].fit(X_array[5], y_array[5]) #distorted input, distorted output, right window


    ########
    #THIS MIGHT NOT WORK, YOU MIGHT NOT BE INDEXING THE WINDOWS CORRECTLY WHEN RESTORING THE DATA
    ########
    for t in range(window_size, height - window_size)
        for i in range(window_size):
            #training on clean input
            X_list_array[0].append(clean[t+i][c]) #clean input, left window
            if (i < window_size / 2):
                X_list_array[1].append(clean[t+i][c]) #clean input, center window
            else:
                X_list_array[1].append(clean[t+i+1][c]) #clean input, center window
            X_list_array[2].append(clean[t+i+1][c]) #clean input, right window

            #training on distorted input
            X_list_array[3].append(distorted[t+i][c]) #distorted input, left window
            if (i < window_size / 2):
                X_list_array[4].append(clean[t+i][c]) #distorted input, center window
            else:
                X_list_array[4].append(clean[t+i+1][c]) #distorted input, center window
            X_list_array[5].append(clean[t+i+1][c]) #distorted input, right window

        restored_array[0][t+window_size][c] = random_forest_array[0].predict(numpy.asarray([X_list_array[]])) #clean input, clean output, left window
        restored_array[1][t+int(window_size/2)][c] = random_forest_array[1].predict(numpy.asarray([X_list_array[]])) #clean input, clean output, center window
        restored_array[2][t][c] = random_forest_array[2].predict(numpy.asarray([X_list_array[]])) #clean input, clean output, right window
        restored_array[3][t+window_size][c] = random_forest_array[3].predict(numpy.asarray([X_list_array[]])) #distorted input, clean output, left window
        restored_array[4][t+int(window_size/2)][c] = random_forest_array[4].predict(numpy.asarray([X_list_array[]])) #distorted input, clean output, center window
        restored_array[5][t][c] = random_forest_array[5].predict(numpy.asarray([X_list_array[]])) #distorted input, clean output, right window
        restored_array[6][t+window_size][c] = random_forest_array[6].predict(numpy.asarray([X_list_array[]])) #distorted input, distorted output, left window
        restored_array[7][t+int(window_size/2)][c] = random_forest_array[7].predict(numpy.asarray([X_list_array[]])) #distorted input, distorted output, center window
        restored_array[8][t][c] = random_forest_array[8].predict(numpy.asarray([X_list_array[]])) #distorted input, distorted output, right window



# X = numpy.empty((0, 5))
# y = numpy.empty(0)
# for t in range(2000, 2100):
#     #X = numpy.concatenate((X, numpy.asarray([[clean_data[t][0], clean_data[t+1][0], clean_data[t+2][0], clean_data[t+3][0], clean_data[t+4][0]]])))
#     X = numpy.concatenate((X, numpy.asarray([[distorted[t][0], distorted[t+1][0], distorted[t+2][0], distorted[t+3][0], distorted[t+4][0]]])))
#     #X = numpy.concatenate((X, numpy.asarray([[distorted[t][1], distorted[t+1][1], distorted[t+2][1], distorted[t+3][1], distorted[t+4][1]]])))
#     y = numpy.concatenate((y, numpy.asarray([clean_data[t+5][0]])))
#     #y = numpy.concatenate((y, numpy.asarray([distorted[t+5][1]])))
#     #y = numpy.concatenate((y, numpy.asarray([distorted[t+5][0]])))

# regr = RandomForestRegressor(max_depth=7, random_state=1, n_estimators=100)
# regr.fit(X, y)
# # print(regr.feature_importances_)
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