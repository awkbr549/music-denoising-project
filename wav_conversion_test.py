#! /usr/bin/python3

from os import stat
import sys
from numpy.random import randint
from numpy import array
import matplotlib.pyplot as pyplot
from imageio import imwrite
from sklearn.feature_extraction.image import extract_patches_2d
from numpy import mean
from sklearn.decomposition import MiniBatchDictionaryLearning

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
print()
data, channels, samples = get_samples(offset=44, num_channels=num_channels, block_align=block_align, bits_per_sample=bits_per_sample, num_data_bytes=subchunk_2_size, num_samples=num_samples)
print("INFO: Read " + str(samples) + " samples across " + str(channels) + " channels. ")
if (not (samples == num_samples)):
    print("ERROR: Sample count mismatch. Ending... ")
    exit(1)
elif (not (channels == num_channels)):
    print("ERROR: Channel count mismatch. Ending... ")
    exit(1)



###
#Creating an image
###
# print(data[0][100000:100100])
# image_list = []
# for i in range(0, 1):
#     image_list.append(data[0][100000:100100])
# #for i in range(0, 1):
# #    image_list.append(data[1][100000:100100])
# #image = array([data[0][100000:101000], data[1][100000:101000]])
# image = array(image_list)
# print(image.shape)

# # grayscale = ((255 * image) / (2.0**bits_per_sample)).astype("uint8")
# # imwrite("output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_image.png", grayscale)
# # pyplot.imshow(grayscale, cmap=pyplot.cm.gray_r)
# # pyplot.xticks()
# # pyplot.yticks(())
# # pyplot.show()

# quantum = image / (2.0**bits_per_sample)
# patch_size = (1, 7)
# patches = extract_patches_2d(quantum, patch_size)
# dictionary = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
# V = dictionary.fit(patches).components_
# exit(0)





###
#Attempting to lower the audio by 1 octave
###
#print("INFO: Lowering audio by one octave... ")
#f = open("output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_lowered.wav", "wb")
#print("INFO: Raising audio by one octave... ")
f = open("output/" + sys.argv[1][sys.argv[1].index("input/") + 6:sys.argv[1].index(".wav")] + "_noised.wav", "wb")
#RIFF chunk descriptor
for c in chunk_id:
    f.write(c.encode())
#write_little_endian(file=f, value=int(36 + (2 * subchunk_2_size)), bits_per_sample=32, start=4, stop=8)
write_little_endian(file=f, value=int(36 + (subchunk_2_size)), bits_per_sample=32, start=4, stop=8)
for c in format_val:
    f.write(c.encode())

#WAV "fmt" sub-chunk
for c in subchunk_1_id:
    f.write(c.encode())
write_little_endian(file=f, value=subchunk_1_size, bits_per_sample=32, start=16, stop=20)
write_little_endian(file=f, value=audio_format, bits_per_sample=16, start=20, stop=22)
write_little_endian(file=f, value=num_channels, bits_per_sample=16, start=22, stop=24)
write_little_endian(file=f, value=sample_rate, bits_per_sample=32, start=24, stop=28)
write_little_endian(file=f, value=byte_rate, bits_per_sample=32, start=28, stop=32)
write_little_endian(file=f, value=block_align, bits_per_sample=16, start=32, stop=34)
write_little_endian(file=f, value=bits_per_sample, bits_per_sample=16, start=34, stop=36)

#WAV "data" sub-chunk
for c in subchunk_2_id:
    f.write(c.encode())
#write_little_endian(file=f, value=int(2 * subchunk_2_size), bits_per_sample=32, start=40, stop=44)
write_little_endian(file=f, value=int(subchunk_2_size), bits_per_sample=32, start=40, stop=44)
bool_help = True
for s in range(0, samples):
#for s in range(0, 100):
    # if (bool_help):
    for c in range(0, channels):
        temp = data[c][s]

        #random noise everything
        # temp = int(data[c][s] + randint(-10000, 10000))
        # if (temp > (2**bits_per_sample) - 1):
        #     temp = (2**bits_per_sample) - 1
        # elif (temp < 0):
        #     temp = 0

        #random noise sometimes
        # temp = data[c][s]
        # if (randint(0, 1000) < 1):
        #     temp = randint(0, (2**bits_per_sample) - 1)
        
        write_little_endian(file=f, value=temp, bits_per_sample=bits_per_sample)
    # bool_help = not bool_help
    # for c in range(0, channels):
    #     write_little_endian(file=f, value=data[c][s], bits_per_sample=bits_per_sample)

f.close()

exit()