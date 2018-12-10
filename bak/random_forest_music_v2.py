#! /usr/bin/env python3

import sys
import soundfile
import numpy
import gc
from itertools import product as cartesian
from psutil import virtual_memory
from os import mkdir
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor

input_file_name = sys.argv[1]
# window_size = int(sys.argv[2])
# if (window_size % 2 == 1):
#     window_size += 1
# MAX_DEPTH = int(sys.argv[3]) #16
# RANDOM_STATE = int(sys.argv[4]) #2
# N_ESTIMATORS = int(sys.argv[5]) #100

output_file_name_base_str = "output/" + input_file_name[input_file_name.rfind("/")+1:input_file_name.index(".wav")] + "/"
try:
    mkdir(path=output_file_name_base_str)
except (FileExistsError):
    pass

window_sizes = [int(sys.argv[2])] #[16] #[4, 8, 16] #[int(sys.argv[2])] #[4, 8, 16]
max_depth_sizes = [int(sys.argv[3])] #[16] #[8, 16] #[int(sys.argv[3])] #[16, 4, 8, 32]
random_state_sizes = [0] #[2, 1, 0]
num_estimators_sizes = [int(sys.argv[4])] #[100, 50, 200, 500]
for window_size, MAX_DEPTH, RANDOM_STATE, N_ESTIMATORS in cartesian(window_sizes, max_depth_sizes, random_state_sizes, num_estimators_sizes):
    print("\n\n-----NEW TRIAL-----")
    print((window_size, MAX_DEPTH, RANDOM_STATE, N_ESTIMATORS))
    gc.collect()

    output_file_name_restored = output_file_name_base_str + "w_" + str(window_size) + "_c_di_do_md_" + str(MAX_DEPTH) + "_rs_" + str(RANDOM_STATE) + "_ne_" + str(N_ESTIMATORS) + ".wav" #distorted input, distorted output, center window
    #print(output_file_name_restored)
    # c = 0
    # print(output_file_name_restored[:output_file_name_restored.index(".wav")] + "_chan_" + str(c) + ".rf")
    # exit()

    print("Extracting data... ", end="")
    sys.stdout.flush()
    clean_data, sample_rate = soundfile.read(input_file_name)
    #clean_data = clean_data[int(30.0*sample_rate):int(40.0*sample_rate)]
    distorted = clean_data.copy()
    height, width = distorted.shape
    print("Done.")

    print("Training forests and reconstruting audio... ")
    sys.stdout.flush()
    restored = numpy.empty((height, width))

    for c in range(width):
        random_forest = RandomForestRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS) #distorted input, distorted output, center window
        random_forest_trained = False
        try:
            random_forest = joblib.load(output_file_name_restored[:output_file_name_restored.index(".wav")] + "_chan_" + str(c) + ".rf")
            random_forest_trained = True
        except (FileNotFoundError):
            pass

        X = numpy.empty((0, window_size)) #distorted input, center window
        y = numpy.empty(0) #distorted output, center window

        X_list = []
        y_list = []

        print("\tExtracting training data for channel " + str(c) + "... ")
        print("|0\t|10\t|20\t|30\t|40\t|50\t|60\t|70\t|80\t|90\t|100\t")
        sys.stdout.flush()
        MAX = height - window_size
        DIVISOR = int(MAX / 80)
        for t in range(MAX):
            #this is just part of the loading bar
            if (t % DIVISOR == 0):
                print("|", end="")
                sys.stdout.flush()

            if (float(virtual_memory().percent) > 85.0):
                print("-----ERROR-----")
                print(" Memory overload. Ending process...")
                exit(1)

            X_list.append([])

            for i in range(window_size):
                if (i < window_size / 2):
                    X_list[t].append(distorted[t+i][c]) #distorted input, center window
                else:
                    X_list[t].append(distorted[t+i+1][c]) #distorted input, center window

            y_list.append((distorted[t+int(window_size/2)][c])) #distorted output, center window
        
        #this is just part of the loading bar
        print("|", end="")
        print("\tDone.")
        sys.stdout.flush()

        X = numpy.concatenate((X, numpy.asarray(X_list))) #distorted input, center window
        X_list = None
        gc.collect()

        y = numpy.concatenate((y, numpy.asarray(y_list))) #distorted output, center window
        y_list = None
        gc.collect() 

        print("\tTraining forest for channel " + str(c) + "... ", end="")
        sys.stdout.flush()
        if (not random_forest_trained):
            random_forest.fit(X, y) #distorted input, distorted output, center window
            joblib.dump(random_forest, output_file_name_restored[:output_file_name_restored.index(".wav")] + "_chan_" + str(c) + ".rf")
        print("Done.")

        print("\tReconstructing audio for channel " + str(c) + "... ")
        print("|0\t|10\t|20\t|30\t|40\t|50\t|60\t|70\t|80\t|90\t|100\t")
        sys.stdout.flush()
        for t in range(window_size, MAX):
            #this is just part of the loading bar
            if (t % DIVISOR == 0):
                print("|", end="")
                sys.stdout.flush()

            restored[t+int(window_size/2)][c] = random_forest.predict([X[t]]) #distorted input, distorted output, center window

        #this is just part of the loading bar
        print("|", end="")
        print("\tDone.")

        #saving each channel in case the program crashes or something
        soundfile.write(output_file_name_restored, restored, sample_rate)

    print("Done.")

    print("Saving reconstructed audio to file... ", end="")
    sys.stdout.flush()
    soundfile.write(output_file_name_restored, restored, sample_rate)
    print("Done.")