#! /usr/bin/env python3

USING_LEFT = False
USING_CENTER = True
USING_RIGHT = False

USING_CLEAN_INPUT = False
USING_DISTORTED_INPUT = True

USING_CLEAN_OUTPUT = False
USING_DISTORTED_OUTPUT = True

import sys
from playsound import playsound
import soundfile
import numpy
from sklearn.externals import joblib
#from sklearn.feature_extraction.image import extract_patches_2d
import gc
from time import sleep
from psutil import virtual_memory
from os import mkdir

from sklearn.ensemble import RandomForestRegressor

input_file_name = sys.argv[1]
window_size = int(sys.argv[2])
#if window_size isn't even, round up
if (window_size % 2 == 1):
    window_size += 1

MAX_DEPTH = sys.argv[3] #16
RANDOM_STATE = sys.argv[4] #2
N_ESTIMATORS = sys.argv[5] #100

#output_file_name_white = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_white_noise.wav"
#output_file_name_base_str = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")]
# output_file_name_restored = [
    # output_file_name_base_str + "_rf_ci_co_l_" + str(window_size) + ".wav", #clean input, clean output, left window
    # output_file_name_base_str + "_rf_ci_co_c_" + str(window_size) + ".wav", #clean input, clean output, center window
    # output_file_name_base_str + "_rf_ci_co_r_" + str(window_size) + ".wav", #clean input, clean output, right window
    # output_file_name_base_str + "_rf_di_co_l_" + str(window_size) + ".wav", #distorted input, clean output, left window
    # output_file_name_base_str + "_rf_di_co_c_" + str(window_size) + ".wav", #distorted input, clean output, center window
    # output_file_name_base_str + "_rf_di_co_r_" + str(window_size) + ".wav", #distorted input, clean output, right window
    # output_file_name_base_str + "_rf_di_do_l_" + str(window_size) + ".wav", #distorted input, distorted output, left window
    # output_file_name_base_str + "_rf_di_do_c_" + str(window_size) + ".wav", #distorted input, distorted output, center window
    # output_file_name_base_str + "_rf_di_do_r_" + str(window_size) + ".wav" #distorted input, distorted output, right window       
    # ]

output_file_name_base_str = "output/" + input_file_name[input_file_name.rfind("/")+1:input_file_name.index(".wav")] + "/"
#mkdir(path=output_file_name_base_str)
# print(output_file_name_base_str)
# exit()
c = 0
output_file_name_restored = output_file_name_base_str + "w_" + str(window_size) + "_c_di_do_md_" + str(MAX_DEPTH) + "_rs_" + str(RANDOM_STATE) + "_ne_" + str(N_ESTIMATORS) + ".wav" #distorted input, distorted output, center window
print(output_file_name_restored)
print(output_file_name_restored[:output_file_name_restored.index(".wav")] + "_chan_" + str(c) + ".rf")
exit()

#output_file_name_random_forest_left = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_random_forest_left_" + str(window_size) + ".wav"
#output_file_name_random_forest_center = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_random_forest_center_" + str(window_size) + ".wav"
#output_file_name_random_forest_right = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_random_forest_right" + str(window_size) + ".wav"


print("Extracting data... ", end="")
sys.stdout.flush()
clean_data, sample_rate = soundfile.read(input_file_name) #returns float64 encoding
print("Done.")

# print("Original file playing... ", end="")
# sys.stdout.flush()
# playsound(input_file_name)
# print("Done.")

#if (sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")].startswith("Arapaho")):
#####
# For data without clean baseline, don't noise to the audio
#####
print()
print("-----WARNING-----")
print("This test is running without adding noise!")
sys.stdout.flush()
distorted = clean_data.copy()
height, width = distorted.shape

# else:
#     #####
#     # For clean data only
#     #####
#     print("Noising data with white noise... ", end="")
#     sys.stdout.flush()
#     distorted = clean_data.copy()
#     height, width = distorted.shape
#     white_noise = 0.005 * numpy.random.randn(height, width) #returns matrix with 0 mean and 1 variance
#     white_noise = 0.03 * numpy.random.randn(height, width) #returns matrix with 0 mean and 1 variance
#     soundfile.write("output/white_noise.wav", white_noise, sample_rate)
#     distorted += white_noise
#     soundfile.write(output_file_name_white, distorted, sample_rate)
#     print("Done.")


# print("Playing file with white noise added... ", end="")
# sys.stdout.flush()
# playsound(output_file_name_white)
# print("Done.")


print("Training forests and reconstructing audio... ")#, end="")
sys.stdout.flush()
# restored_array = [
#     numpy.empty((height, width)), #clean input, clean output, left window
#     numpy.empty((height, width)), #clean input, clean output, center window
#     numpy.empty((height, width)), #clean input, clean output, right window
#     numpy.empty((height, width)), #distorted input, clean output, left window
#     numpy.empty((height, width)), #distorted input, clean output, center window
#     numpy.empty((height, width)), #distorted input, clean output, right window
#     numpy.empty((height, width)), #distorted input, distorted output, left window
#     numpy.empty((height, width)), #distorted input, distorted output, center window
#     numpy.empty((height, width)) #distorted input, distorted output, right window
#     ]
restored = numpy.empty((height, width))

for c in range(width):



    #####
    #TESTING PURPOSES ONLY
    #####
    #c = 1



    random_forest_array = [
        None, #RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #clean input, clean output, left window
        None, #RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #clean input, clean output, center window
        None, #RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #clean input, clean output, right window
        None, #RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #distorted input, clean output, left window
        None, #RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #distorted input, clean output, center window
        None, #RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #distorted input, clean output, right window
        None, #RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #distorted input, distorted output, left window
        RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS), #distorted input, distorted output, center window
        None #RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS) #distorted input, distorted output, right window
        ]
    random_forest_array_trained = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        ]

    for i in range(len(random_forest_array)):
        try:
            #random_forest_array[i] = joblib.load("output/forest_c_" + str(c) + "_md_" + str(MAX_DEPTH) + "_rs_" + str(RANDOM_STATE) + "_ne_" + str(N_ESTIMATORS) + "_" + output_file_name_restored[i][7:output_file_name_restored[i].find(".wav")] + ".rf")
            random_forest_array[i] = joblib.load(output_file_name_restored[:output_file_name_restored.index(".wav")] + "_c_" + str(c) + ".rf")
            random_forest_array_trained[i] = True
        except (FileNotFoundError):
            pass
            #random_forest_array[i] = RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS)

    X_array = [
        numpy.empty((0, window_size)), #clean input, left window
        numpy.empty((0, window_size)), #clean input, center window
        numpy.empty((0, window_size)), #clean input, right window
        numpy.empty((0, window_size)), #distorted input, left window
        numpy.empty((0, window_size)), #distorted input, center window
        numpy.empty((0, window_size)) #distorted input, right window
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

    print("\tExtracting training data for channel " + str(c) + "... ", end="")
    sys.stdout.flush()
    MAX = height - window_size#400000 #height - window_size
    print("\n|0\t|10\t|20\t|30\t|40\t|50\t|60\t|70\t|80\t|90\t|100\t")
    sys.stdout.flush()
    DIVISOR = int(MAX / 80)
    for t in range(MAX):        
        #this is just part of the loading bar
        if (t % DIVISOR == 0):
            print("|", end="")
            sys.stdout.flush()

        if (float(virtual_memory().percent) > 85.0):
            print("-----ERROR-----")
            print(" Memory overload. Ending process...")
            exit()

        for X_list in X_list_array:
            X_list.append([])
        for i in range(window_size):
            if (USING_CLEAN_INPUT):
                #training on clean input
                if (USING_LEFT):
                    X_list_array[0][t].append(clean_data[t+i][c]) #clean input, left window
                if (USING_CENTER):
                    if (i < window_size / 2):
                        X_list_array[1][t].append(clean_data[t+i][c]) #clean input, center window
                    else:
                        X_list_array[1][t].append(clean_data[t+i+1][c]) #clean input, center window
                if (USING_RIGHT):
                    X_list_array[2][t].append(clean_data[t+i+1][c]) #clean input, right window

            if (USING_DISTORTED_INPUT):
                #training on distorted input
                if (USING_LEFT):
                    X_list_array[3][t].append(distorted[t+i][c]) #distorted input, left window
                if (USING_CENTER):
                    if (i < window_size / 2):
                        X_list_array[4][t].append(distorted[t+i][c]) #distorted input, center window
                    else:
                        X_list_array[4][t].append(distorted[t+i+1][c]) #distorted input, center window
                if (USING_RIGHT):
                    X_list_array[5][t].append(distorted[t+i+1][c]) #distorted input, right window

        if (USING_CLEAN_OUTPUT):
            #training on clean output
            if (USING_LEFT):
                y_list_array[0].append((clean_data[t+window_size][c])) #clean output, left window
            if (USING_CENTER):
                y_list_array[1].append((clean_data[t+int(window_size/2)][c])) #clean output, center window
            if (USING_RIGHT):
                y_list_array[2].append((clean_data[t][c])) #clean output, right window

        if (USING_DISTORTED_OUTPUT):
            #training on distorted output
            if (USING_LEFT):
                y_list_array[3].append((distorted[t+window_size][c])) #distorted output, left window
            if (USING_CENTER):
                y_list_array[4].append((distorted[t+int(window_size/2)][c])) #distorted output, center window
            if (USING_RIGHT):
                y_list_array[5].append((distorted[t][c])) #distorted output, right window

    #this is just part of the loading bar
    print("|", end="")
    print("\tDone.")

    if (USING_CLEAN_INPUT):
        if (USING_LEFT):
            X_array[0] = numpy.concatenate((X_array[0], numpy.asarray(X_list_array[0]))) #clean input, left window
            X_list_array[0] = None
            gc.collect()
        if (USING_CENTER):
            X_array[1] = numpy.concatenate((X_array[1], numpy.asarray(X_list_array[1]))) #clean input, center window
            X_list_array[1] = None
            gc.collect()
        if (USING_RIGHT):
            X_array[2] = numpy.concatenate((X_array[2], numpy.asarray(X_list_array[2]))) #clean input, right window
            X_list_array[2] = None
            gc.collect()

    if (USING_DISTORTED_INPUT):
        if (USING_LEFT):
            X_array[3] = numpy.concatenate((X_array[3], numpy.asarray(X_list_array[3]))) #distorted input, left window
            X_list_array[3] = None
            gc.collect()
        if (USING_CENTER):
            X_array[4] = numpy.concatenate((X_array[4], numpy.asarray(X_list_array[4]))) #distorted input, center window
            X_list_array[4] = None
            gc.collect()
        if (USING_RIGHT):
            X_array[5] = numpy.concatenate((X_array[5], numpy.asarray(X_list_array[5]))) #distorted input, right window
            X_list_array[5] = None
            gc.collect()



    if (USING_CLEAN_OUTPUT):
        if (USING_LEFT):
            y_array[0] = numpy.concatenate((y_array[0], numpy.asarray(y_list_array[0]))) #clean output, left window
            y_list_array[0] = None
            gc.collect()
        if (USING_CENTER):
            y_array[1] = numpy.concatenate((y_array[1], numpy.asarray(y_list_array[1]))) #clean output, center window
            y_list_array[1] = None
            gc.collect()
        if (USING_RIGHT):
            y_array[2] = numpy.concatenate((y_array[2], numpy.asarray(y_list_array[2]))) #clean output, right window
            y_list_array[2] = None
            gc.collect()

    if (USING_DISTORTED_OUTPUT):
        if (USING_LEFT):
            y_array[3] = numpy.concatenate((y_array[3], numpy.asarray(y_list_array[3]))) #distorted output, left window
            y_list_array[3] = None
            gc.collect()
        if (USING_CENTER):
            y_array[4] = numpy.concatenate((y_array[4], numpy.asarray(y_list_array[4]))) #distorted output, center window
            y_list_array[4] = None
            gc.collect()
        if (USING_RIGHT):
            y_array[5] = numpy.concatenate((y_array[5], numpy.asarray(y_list_array[5]))) #distorted output, right window
            y_list_array[5] = None
            gc.collect()

    
    print("\tDone.")

    print("\tTraining forests for channel " + str(c) + "... ")#, end="")
    sys.stdout.flush()

    if (USING_CLEAN_INPUT and USING_CLEAN_OUTPUT and USING_LEFT):
        if (not random_forest_array_trained[0]):
            random_forest_array[0].fit(X_array[0], y_array[0]) #clean input, clean output, left window
            joblib.dump(random_forest_array[0], "output/forest_c_" + str(c) + "_md_" + str(MAX_DEPTH) + "_rs_" + str(RANDOM_STATE) + "_ne_" + str(N_ESTIMATORS) + "_" + output_file_name_restored[0][7:output_file_name_restored[i].find(".wav")] + ".rf")
        print("\t\tTrained forest 0.")
        sys.stdout.flush()

    if (USING_CLEAN_INPUT and USING_CLEAN_OUTPUT and USING_CENTER):
        if (not random_forest_array_trained[1]):
            random_forest_array[1].fit(X_array[1], y_array[1]) #clean input, clean output, center window
            joblib.dump(random_forest_array[1], "output/forest_c_" + str(c) + "_md_" + str(MAX_DEPTH) + "_rs_" + str(RANDOM_STATE) + "_ne_" + str(N_ESTIMATORS) + "_" + output_file_name_restored[1][7:output_file_name_restored[i].find(".wav")] + ".rf")
        print("\t\tTrained forest 1.")
        sys.stdout.flush()

    if (USING_CLEAN_INPUT and USING_CLEAN_OUTPUT and USING_RIGHT):
        if (not random_forest_array_trained[2]):
            random_forest_array[2].fit(X_array[2], y_array[2]) #clean input, clean output, right window
            joblib.dump(random_forest_array[2], "output/forest_c_" + str(c) + "_md_" + str(MAX_DEPTH) + "_rs_" + str(RANDOM_STATE) + "_ne_" + str(N_ESTIMATORS) + "_" + output_file_name_restored[2][7:output_file_name_restored[i].find(".wav")] + ".rf")
        print("\t\tTrained forest 2.")
        sys.stdout.flush()

    if (USING_DISTORTED_INPUT and USING_CLEAN_OUTPUT and USING_LEFT):
        if (not random_forest_array_trained[3]):
            random_forest_array[3].fit(X_array[3], y_array[0]) #distorted input, clean output, left window
            joblib.dump(random_forest_array[3], "output/forest_c_" + str(c) + "_md_" + str(MAX_DEPTH) + "_rs_" + str(RANDOM_STATE) + "_ne_" + str(N_ESTIMATORS) + "_" + output_file_name_restored[3][7:output_file_name_restored[i].find(".wav")] + ".rf")
        print("\t\tTrained forest 3.")
        sys.stdout.flush()

    if (USING_DISTORTED_INPUT and USING_CLEAN_OUTPUT and USING_CENTER):
        if (not random_forest_array_trained[4]):
            random_forest_array[4].fit(X_array[4], y_array[1]) #distorted input, clean output, center window
            joblib.dump(random_forest_array[4], "output/forest_c_" + str(c) + "_md_" + str(MAX_DEPTH) + "_rs_" + str(RANDOM_STATE) + "_ne_" + str(N_ESTIMATORS) + "_" + output_file_name_restored[4][7:output_file_name_restored[4].find(".wav")] + ".rf")
        print("\t\tTrained forest 4.")
        sys.stdout.flush()

    if (USING_DISTORTED_INPUT and USING_CLEAN_OUTPUT and USING_RIGHT):
        if (not random_forest_array_trained[5]):
            random_forest_array[5].fit(X_array[5], y_array[2]) #distorted input, clean output, right window
            joblib.dump(random_forest_array[5], "output/forest_c_" + str(c) + "_md_" + str(MAX_DEPTH) + "_rs_" + str(RANDOM_STATE) + "_ne_" + str(N_ESTIMATORS) + "_" + output_file_name_restored[5][7:output_file_name_restored[i].find(".wav")] + ".rf")
        print("\t\tTrained forest 5.")
        sys.stdout.flush()

    if (USING_DISTORTED_INPUT and USING_DISTORTED_OUTPUT and USING_LEFT):
        if (not random_forest_array_trained[6]):
            random_forest_array[6].fit(X_array[3], y_array[3]) #distorted input, distorted output, left window
            joblib.dump(random_forest_array[6], "output/forest_c_" + str(c) + "_md_" + str(MAX_DEPTH) + "_rs_" + str(RANDOM_STATE) + "_ne_" + str(N_ESTIMATORS) + "_" + output_file_name_restored[6][7:output_file_name_restored[i].find(".wav")] + ".rf")
        print("\t\tTrained forest 6.")
        sys.stdout.flush()

    if (USING_DISTORTED_INPUT and USING_DISTORTED_OUTPUT and USING_CENTER):
        if (not random_forest_array_trained[7]):
            random_forest_array[7].fit(X_array[4], y_array[4]) #distorted input, distorted output, center window
            joblib.dump(random_forest_array[7], "output/forest_c_" + str(c) + "_md_" + str(MAX_DEPTH) + "_rs_" + str(RANDOM_STATE) + "_ne_" + str(N_ESTIMATORS) + "_" + output_file_name_restored[7][7:output_file_name_restored[i].find(".wav")] + ".rf")
        print("\t\tTrained forest 7.")
        sys.stdout.flush()

    if (USING_DISTORTED_INPUT and USING_DISTORTED_OUTPUT and USING_RIGHT):
        if (not random_forest_array_trained[8]):
            random_forest_array[8].fit(X_array[5], y_array[5]) #distorted input, distorted output, right window
            joblib.dump(random_forest_array[8], "output/forest_c_" + str(c) + "_md_" + str(MAX_DEPTH) + "_rs_" + str(RANDOM_STATE) + "_ne_" + str(N_ESTIMATORS) + "_" + output_file_name_restored[8][7:output_file_name_restored[i].find(".wav")] + ".rf")
        print("\t\tTrained forest 8.")
        sys.stdout.flush()

    # print("\tDone.")

    print("\tReconstructing audio for channel " + str(c) + "... ", end="")
    print("\n|0\t|10\t|20\t|30\t|40\t|50\t|60\t|70\t|80\t|90\t|100\t")
    sys.stdout.flush()
    DIVISOR = int(MAX / 80)
    for t in range(window_size, MAX):        
        #this is just part of the loading bar
        if (t % DIVISOR == 0):
            print("|", end="")
            sys.stdout.flush()

        if (USING_CLEAN_INPUT and USING_CLEAN_OUTPUT and USING_LEFT):
            restored_array[0][t+window_size][c] = random_forest_array[0].predict([X_array[0][t]]) #clean input, clean output, left window
        if (USING_CLEAN_INPUT and USING_CLEAN_OUTPUT and USING_CENTER):
            restored_array[1][t+int(window_size/2)][c] = random_forest_array[1].predict([X_array[1][t]]) #clean input, clean output, center window
        if (USING_CLEAN_INPUT and USING_CLEAN_OUTPUT and USING_RIGHT):
            restored_array[2][t][c] = random_forest_array[2].predict([X_array[2][t]]) #clean input, clean output, right window
        if (USING_DISTORTED_INPUT and USING_CLEAN_OUTPUT and USING_LEFT):
            restored_array[3][t+window_size][c] = random_forest_array[3].predict([X_array[3][t]]) #distorted input, clean output, left window
        if (USING_DISTORTED_INPUT and USING_CLEAN_OUTPUT and USING_CENTER):
            restored_array[4][t+int(window_size/2)][c] = random_forest_array[4].predict([X_array[4][t]]) #distorted input, clean output, center window
        if (USING_DISTORTED_INPUT and USING_CLEAN_OUTPUT and USING_RIGHT):
            restored_array[5][t][c] = random_forest_array[5].predict([X_array[5][t]]) #distorted input, clean output, right window
        if (USING_DISTORTED_INPUT and USING_DISTORTED_OUTPUT and USING_LEFT):
            restored_array[6][t+window_size][c] = random_forest_array[6].predict([X_array[3][t]]) #distorted input, distorted output, left window
        if (USING_DISTORTED_INPUT and USING_DISTORTED_OUTPUT and USING_CENTER):
            restored_array[7][t+int(window_size/2)][c] = random_forest_array[7].predict([X_array[4][t]]) #distorted input, distorted output, center window
        if (USING_DISTORTED_INPUT and USING_DISTORTED_OUTPUT and USING_RIGHT):
            restored_array[8][t][c] = random_forest_array[8].predict([X_array[5][t]]) #distorted input, distorted output, right window
    
    #this is just part of the loading bar
    print("|", end="")
    print("\tDone.")




    #####
    #TESTING PURPOSES ONLY
    #####
    #break




print("Done.")

print("Saving reconstructed audio to files... ", end="")
sys.stdout.flush()
for i in range(len(output_file_name_restored)):
    soundfile.write(output_file_name_restored[i], restored_array[i], sample_rate)

    #####
    #TESTING PURPOSES ONLY
    #####
    #break

print("Done.")