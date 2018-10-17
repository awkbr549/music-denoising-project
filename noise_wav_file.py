#! /usr/bin/python3

from os import stat
import sys
from numpy.random import randint
from numpy import array

TESTING = False

#interpret a little endian number from the byte array
def read_little_endian(start=0, stop=0):
    temp = 0
    for i in range(start, stop):
        temp += (byte_array[i] << ((i - start) * 8))
    return temp

def write_little_endian(file=None, value=0, bits_per_sample=8, start=0, stop=0):
    temp = value
    buff = []
    for _ in range(0, int(bits_per_sample / 8)):
        buff.append(temp & 0xFF)
        temp = temp >> 8
    for i in range(0, len(buff)):
        file.write(bytes([buff[i]]))

def write_big_endian(file=None, value=0, bits_per_sample=32, start=0, stop=0):
    temp = value
    buff = []
    for _ in range(0, int(bits_per_sample / 8)):
        buff.append(temp & 0xFF)
        temp = temp >> 8
    for i in range(0, int(len(buff) / 2)):
        buff[i] = buf[len(buff) - i - 1]
    for i in range(0, len(buff)):
        file.write(bytes([buff[i]]))

def get_samples(offset=44, num_channels=1, block_align=1, bits_per_sample=8, num_data_bytes=0, num_samples=0):
    temp = []
    for _ in range(0, num_channels):
        temp.append([])

    index = offset
    for sample in range(0, num_samples):
    #for sample in range(0, 64):
        for channel in range(0, len(temp)):
            temp[channel].append(read_little_endian(index, index + int(bits_per_sample / 8)))
            index += int(bits_per_sample / 8)
            #print(hex(temp[channel][sample]), end="\t")
        #print()

    return temp, len(temp), len(temp[0])



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
chunk_size = read_little_endian(4, 8)
format_val = ""
for i in range(8, 12):
    format_val += str(chr(byte_array[i]))



###
#WAV "fmt" sub-chunk
###
print()
subchunk_1_id = ""
for i in range(12, 16):
    subchunk_1_id += str(chr(byte_array[i]))
subchunk_1_size = read_little_endian(16, 20)
audio_format = read_little_endian(20, 22)
num_channels = read_little_endian(22, 24)
sample_rate = read_little_endian(24, 28)
byte_rate = read_little_endian(28, 32)
block_align = read_little_endian(32, 34)
bits_per_sample = read_little_endian(34, 36)



###
#WAV "data" sub-chunk
###
print()
subchunk_2_id = ""
for i in range(36, 40):
    subchunk_2_id += str(chr(byte_array[i]))
subchunk_2_size = read_little_endian(40, 44)



###
#Data processing
###
num_samples = int(subchunk_2_size / (num_channels * (bits_per_sample / 8)))
data, channels, samples = get_samples(offset=44, num_channels=num_channels, block_align=block_align, bits_per_sample=bits_per_sample, num_data_bytes=subchunk_2_size, num_samples=num_samples)
print("INFO: Read " + str(samples) + " samples across " + str(channels) + " channels. ")
if (not (samples == num_samples)):
    print("ERROR: Sample count mismatch. Ending... ")
    exit(1)
elif (not (channels == num_channels)):
    print("ERROR: Channel count mismatch. Ending... ")
    exit(1)



if (TESTING):
    print("ChunkId: " + chunk_id)
    print("ChunkSize: " + str(chunk_size))
    print("Format: " + format_val)
    print("Subchunk1Id: " + subchunk_1_id)
    print("Subchunk1Size: " + str(subchunk_1_size))
    print("AudioFormat: " + str(audio_format))
    print("NumChannels: " + str(num_channels))
    print("SampleRate: " + str(sample_rate))
    print("ByteRate: " + str(byte_rate))
    print("BlockAlign: " + str(block_align))
    print("BitsPerSample: " + str(bits_per_sample))
    print("Subchunk2Id: " + subchunk_2_id)
    print("Subchunk2Size: " + str(subchunk_2_size))
    print("NumSamples: " + str(num_samples))
    print()



###
#Opening files
###
crackle_double = float(sys.argv[2])
white_int = int(sys.argv[3])
f_crackle = open("output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_crackle_" + sys.argv[2][:sys.argv[2].index(".")] + "_" + sys.argv[2][sys.argv[2].index(".")+1:] + ".wav", "wb")
f_white = open("output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_white_noise_" + sys.argv[3] + ".wav", "wb")
f_both = open("output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_crackle_" + sys.argv[2][:sys.argv[2].index(".")] + "_white_noise" + sys.argv[2][sys.argv[2].index(".")+1:] + "_" + sys.argv[3] + ".wav", "wb")
#RIFF chunk descriptor
for c in chunk_id:
    f_crackle.write(c.encode())
    f_white.write(c.encode())
    f_both.write(c.encode())

#write_little_endian(file=f, value=int(36 + (2 * subchunk_2_size)), bits_per_sample=32, start=4, stop=8)
write_little_endian(file=f_crackle, value=int(36 + (subchunk_2_size)), bits_per_sample=32, start=4, stop=8)
write_little_endian(file=f_white, value=int(36 + (subchunk_2_size)), bits_per_sample=32, start=4, stop=8)
write_little_endian(file=f_both, value=int(36 + (subchunk_2_size)), bits_per_sample=32, start=4, stop=8)

for c in format_val:
    f_crackle.write(c.encode())
    f_white.write(c.encode())
    f_both.write(c.encode())

#WAV "fmt" sub-chunk
for c in subchunk_1_id:
    f_crackle.write(c.encode())
    f_white.write(c.encode())
    f_both.write(c.encode())

write_little_endian(file=f_crackle, value=subchunk_1_size, bits_per_sample=32, start=16, stop=20)
write_little_endian(file=f_white, value=subchunk_1_size, bits_per_sample=32, start=16, stop=20)
write_little_endian(file=f_both, value=subchunk_1_size, bits_per_sample=32, start=16, stop=20)

write_little_endian(file=f_crackle, value=audio_format, bits_per_sample=16, start=20, stop=22)
write_little_endian(file=f_white, value=audio_format, bits_per_sample=16, start=20, stop=22)
write_little_endian(file=f_both, value=audio_format, bits_per_sample=16, start=20, stop=22)

write_little_endian(file=f_crackle, value=num_channels, bits_per_sample=16, start=22, stop=24)
write_little_endian(file=f_white, value=num_channels, bits_per_sample=16, start=22, stop=24)
write_little_endian(file=f_both, value=num_channels, bits_per_sample=16, start=22, stop=24)

write_little_endian(file=f_crackle, value=sample_rate, bits_per_sample=32, start=24, stop=28)
write_little_endian(file=f_white, value=sample_rate, bits_per_sample=32, start=24, stop=28)
write_little_endian(file=f_both, value=sample_rate, bits_per_sample=32, start=24, stop=28)

write_little_endian(file=f_crackle, value=byte_rate, bits_per_sample=32, start=28, stop=32)
write_little_endian(file=f_white, value=byte_rate, bits_per_sample=32, start=28, stop=32)
write_little_endian(file=f_both, value=byte_rate, bits_per_sample=32, start=28, stop=32)

write_little_endian(file=f_crackle, value=block_align, bits_per_sample=16, start=32, stop=34)
write_little_endian(file=f_white, value=block_align, bits_per_sample=16, start=32, stop=34)
write_little_endian(file=f_both, value=block_align, bits_per_sample=16, start=32, stop=34)

write_little_endian(file=f_crackle, value=bits_per_sample, bits_per_sample=16, start=34, stop=36)
write_little_endian(file=f_white, value=bits_per_sample, bits_per_sample=16, start=34, stop=36)
write_little_endian(file=f_both, value=bits_per_sample, bits_per_sample=16, start=34, stop=36)

#WAV "data" sub-chunk
for c in subchunk_2_id:
    f_crackle.write(c.encode())
    f_white.write(c.encode())
    f_both.write(c.encode())

#write_little_endian(file=f, value=int(2 * subchunk_2_size), bits_per_sample=32, start=40, stop=44)
write_little_endian(file=f_crackle, value=subchunk_2_size, bits_per_sample=32, start=40, stop=44)
write_little_endian(file=f_white, value=subchunk_2_size, bits_per_sample=32, start=40, stop=44)
write_little_endian(file=f_both, value=subchunk_2_size, bits_per_sample=32, start=40, stop=44)


###
#Noising 
###
crackle_inv = int(1.0 / crackle_double)
white_exp = 10**white_int
for s in range(0, samples):
    for c in range(0, channels):
        temp_crackle = temp_white = temp_both = data[c][s]

        #white noise
        temp_rand_int = randint(-white_exp, white_exp)
        temp_white += temp_rand_int
        temp_both += temp_rand_int
        temp_white = max(0, min((2**bits_per_sample) - 1, temp_white))
        temp_both = max(0, min((2**bits_per_sample) - 1, temp_both))
        write_little_endian(file=f_white, value=temp_white, bits_per_sample=bits_per_sample)

        #crackle
        if (randint(1, crackle_inv) == 1):
            temp_crackle = randint(0, (2**bits_per_sample) - 1)
            temp_both = randint(0, (2**bits_per_sample) - 1)
        write_little_endian(file=f_crackle, value=temp_crackle, bits_per_sample=bits_per_sample)
        write_little_endian(file=f_both, value=temp_both, bits_per_sample=bits_per_sample)

f_crackle.close()
f_white.close()
f_both.close()
exit()