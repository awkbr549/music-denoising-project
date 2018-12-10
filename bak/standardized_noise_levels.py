#! /usr/bin/env python3

import sys
import soundfile
import numpy
from itertools import product as cartesian

input_file_name = sys.argv[1]
clean_data, sample_rate = soundfile.read(input_file_name)
output_file_base = input_file_name[input_file_name.index("input/") + 6:input_file_name.index(".wav")]

white_noise_levels = [0.01, 0.005, 0.02, 0.04]
white_noise = [None, None, None, None]
crackle_noise_levels = [1000, 500, 2000, 4000]
crackle_noise = [None, None, None, None]

height, width = clean_data.shape

#white noise only
for i in range(len(white_noise_levels)):
    distorted = clean_data.copy()
    white_noise[i] = white_noise_levels[i] * numpy.random.randn(height, width)
    soundfile.write("output/white_noise_0_" + str(white_noise_levels[i])[2:] + ".wav", white_noise[i], sample_rate)
    distorted += white_noise[i]
    #distorted /= max(abs(distorted.min()), abs(distorted.max()))
    soundfile.write("output/" + output_file_base + "_white_noise_0_" + str(white_noise_levels[i])[2:] + ".wav", distorted, sample_rate)

#crackle only
for i in range(len(crackle_noise_levels)):
    distorted = clean_data.copy()
    for s in range(height):
        for c in range(width):
            if (numpy.random.randint(1, crackle_noise_levels[i] + 1) == 1):
                distorted[s][c] = (numpy.random.ranf() * 2.0) - 1.0
    #distorted /= max(abs(distorted.min()), abs(distorted.max()))
    soundfile.write("output/" + output_file_base + "_crackle_noise_" + str(crackle_noise_levels[i]) + ".wav", distorted, sample_rate)

#white and crackle
for i in range(len(white_noise_levels)):
    distorted = clean_data.copy()
    distorted += white_noise[i]
    for s in range(height):
        for c in range(width):
            if (numpy.random.randint(1, crackle_noise_levels[i] + 1) == 1):
                distorted[s][c] = (numpy.random.ranf() * 2.0) - 1.0
    soundfile.write("output/" + output_file_base + "_white_noise_0_" + str(white_noise_levels[i])[2:] + "_crackle_noise_" + str(crackle_noise_levels[i]) + ".wav", distorted, sample_rate)



# white = white_noise_levels[0]
# crackle = crackle_noise_levels[0]
# distorted = clean_data.copy()
# distorted += white_noise[0]
# for s in range(height):
#     for c in range(width):
#         if (numpy.random.randint(1, crackle + 1) == 1):
#             distorted[s][c] = (numpy.random.ranf() * 2.0) - 1.0
# soundfile.write("output/" + output_file_base + "_white_noise_0_" + str(white)[2:] + "_crackle_noise_" + str(crackle) + ".wav", distorted, sample_rate)
