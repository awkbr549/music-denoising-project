#! /usr/bin/python3

from os import stat
import sys

#interpret a little endian number from the byte array
def read_little_endian(start=0, stop=0):
    temp = 0
    for i in range(start, stop):
        temp += (byte_array[i] << ((i - start) * 8))
    return temp

###
#file size and reading
###
stat_info = stat(sys.argv[1])
f = open(sys.argv[1], "rb")
byte_array = f.read(stat_info.st_size)
f.close()

###
#RIFF chucnk descriptor
###
chunk_id = ""
for i in range(0, 4):
    chunk_id += str(chr(byte_array[i]))
print("ChunkId: " + chunk_id)

chunk_size = read_little_endian(4, 8)
print("ChunkSize: " + str(chunk_size))

format_val = ""
for i in range(8, 12):
    format_val += str(chr(byte_array[i]))
print("Format: " + format_val)



###
#WAV "fmt" sub-chunk
###
print()
subchunk_1_id = ""
for i in range(12, 16):
    subchunk_1_id += str(chr(byte_array[i]))
print("Subchunk1Id: " + subchunk_1_id)

subchunk_1_size = read_little_endian(16, 20)
print("Subchunk1Size: " + str(subchunk_1_size))

audio_format = read_little_endian(20, 22)
print("AudioFormat: " + str(audio_format))

num_channels = read_little_endian(22, 24)
print("NumChannels: " + str(num_channels))

sample_rate = read_little_endian(24, 28)
print("SampleRate: " + str(sample_rate))

byte_rate = read_little_endian(28, 32)
print("ByteRate: " + str(byte_rate))

block_align = read_little_endian(32, 34)
print("BlockAlign: " + str(block_align))

bits_per_sample = read_little_endian(34, 36)
print("BitsPerSample: " + str(bits_per_sample))



###
#WAV "data" sub-chunk
###
print()
subchunk_2_id = ""
for i in range(36, 40):
    subchunk_2_id += str(chr(byte_array[i]))
print("Subchunk2Id: " + subchunk_2_id)

subchunk_2_size = read_little_endian(40, 44)
print("Subchunk2Size: " + str(subchunk_2_size))



###
#Data processing
###
num_samples = int(subchunk_2_size / (num_channels * (bits_per_sample / 8)))
print("NumSamples: " + str(num_samples))

exit()