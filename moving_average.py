#! /usr/bin/env python3

import sys
from playsound import playsound
import soundfile
import numpy

input_file_name = sys.argv[1]
window_size = int(sys.argv[2])
#if window_size isn't even, round up
if (window_size % 2 == 1):
    window_size += 1

output_file_name_white = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_white_noise.wav"
output_file_name_moving_avg_left = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_moving_avg_left_" + str(window_size) + ".wav"
output_file_name_moving_avg_center = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_moving_avg_center_" + str(window_size) + ".wav"
output_file_name_moving_avg_right = "output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_moving_avg_right_" + str(window_size) + ".wav"

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

restored_left = numpy.empty((height, width))
restored_center = numpy.empty((height, width))
restored_right = numpy.empty((height, width))
print("Applying moving average for left, centered, and right windows... ", end="")
sys.stdout.flush()

temp_avg_diff_left = 0
temp_avg_diff_center = 0
temp_avg_diff_right = 0

for c in range(width):
    for t in range(height - window_size):
        #temp_avg = 0
        avg_list = []
        for i in range(window_size+1):
            #print(t+i)
            avg_list.append(distorted[t+i][c])
            #temp_avg += distorted[t+i][c]
        avg_val = numpy.mean(avg_list)
        #avg_val = temp_avg / (window_size + 1)

        restored_left[t+window_size][c] = avg_val
        restored_center[t+int(window_size/2)][c] = avg_val
        restored_right[t][c] = avg_val
        temp_avg_diff_left += (clean_data[t+window_size][c] - avg_val)
        temp_avg_diff_center += (clean_data[t+int(window_size/2)][c] - avg_val)
        temp_avg_diff_right += (clean_data[t][c] - avg_val)
soundfile.write(output_file_name_moving_avg_left, restored_left, sample_rate)
soundfile.write(output_file_name_moving_avg_center, restored_center, sample_rate)
soundfile.write(output_file_name_moving_avg_right, restored_right, sample_rate)
print("Done.")

print("Statistics: ")
print("\tAverage difference for left window:\t" + str(temp_avg_diff_left / (height - window_size)))
print("\tAverage difference for centered window:\t" + str(temp_avg_diff_center / (height - window_size)))
print("\tAverage difference for right window:\t" + str(temp_avg_diff_right / (height - window_size)))

# print("Playing file after moving average with left window... ", end="")
# sys.stdout.flush()
# playsound(output_file_name_moving_avg_left)
# print("Done.")

# print("Playing file after moving average with center window... ", end="")
# sys.stdout.flush()
# playsound(output_file_name_moving_avg_center)
# print("Done.")

# print("Playing file after moving average with right window... ", end="")
# sys.stdout.flush()
# playsound(output_file_name_moving_avg_right)
# print("Done.")
