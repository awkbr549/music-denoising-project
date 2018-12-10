#! /usr/bin/env python3

import sys
import soundfile
import numpy
import gc
from os import mkdir

input_file_name = sys.argv[1]
output_file_name_base_str = "output/" + input_file_name[input_file_name.rfind("/")+1:input_file_name.index(".wav")] + "/"
try:
    mkdir(path=output_file_name_base_str)
except (FileExistsError):
    pass

#still to do (8, 32, 200) and (16, 32, 200) from Random Forest
window_sizes = [int(sys.argv[2])] #[2, 4, 8, 16, 32]
for window_size in window_sizes:
    print("\n\n-----NEW TRIAL-----")
    print((window_size))
    gc.collect()

    output_file_name_restored = output_file_name_base_str + "mov_avg_w_" + str(window_size) + ".wav" #center window

    print("Extracting data... ", end="")
    sys.stdout.flush()
    clean_data, sample_rate = soundfile.read(input_file_name)
    clean_data = clean_data[int(30.0*sample_rate):int(40.0*sample_rate)]
    distorted = clean_data.copy()
    height, width = distorted.shape
    print("Done.")

    print("Applying moving average for window size " + str(window_size) + "... ", end="")
    sys.stdout.flush()
    restored = numpy.empty((height, width))

    for c in range(width):
        for t in range(height - window_size):
            avg_list = []
            for i in range(window_size+1):
                avg_list.append(distorted[t+i][c])
            avg_val = numpy.mean(avg_list)
            restored[t+int(window_size/2)][c] = avg_val
    soundfile.write(output_file_name_restored, restored, sample_rate)
    print("Done.")
print("Done.")