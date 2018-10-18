#!/usr/bin/env python3
# coding: utf-8

# In[1]:


from os import stat
import sys
from platform import system

import scipy
import numpy as np
import matplotlib.pyplot as plt
import wave
import soundfile as sf

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
 
from playsound import playsound
from scipy.io import wavfile


# In[2]:


def plotFile(filename):
    #credit to http://gree2.github.io/python/2015/04/14/python-code-read-wave-file-and-plot
    wav_file = wave.open(filename,'rb')
    params = wav_file.getparams()
    
    nchannels, sampwidth, framerate, nframes = params[:4]

    str_data = wav_file.readframes(nframes)
    wav_file.close()
    
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data.shape = -1,2
    wave_data = wave_data.T
    time = np.arange(0,nframes)*(1.0/framerate)

    titlech1 = filename + ' channel 1'
    titlech2 = filename + ' channel 2'
    
    duration = nframes/float(framerate)
    xticks = np.arange(0, duration, 2)
    plt.figure(figsize=(12,6))
    plt.subplot(211).set_xticks(xticks)
    plt.plot(time, wave_data[0])
    plt.title(titlech1, loc='left')
    
    plt.subplot(212).set_xticks(xticks)
    plt.plot(time, wave_data[1], c="g")
    plt.xlabel("time (seconds)")
    plt.title(titlech2, loc='left')
    plt.show()
    
    playsound(filename)
    return


# In[9]:

this_os = system()
if (this_os == "Linux"):
    playsound('./input/Alarm01.wav')
else:
    playsound('D:\Documents-data\Advanced Data Analytics\Group Project\music-denoising-project\input\Alarm01.wav')


# In[10]:

filename = ""
if (this_os == "Linux"):
    filename = './input/Alarm01.wav'
else:
    filename='D:\Documents-data\Advanced Data Analytics\Group Project\music-denoising-project\input\Alarm01.wav'
#filename='music-denoising-project\input\Ring01.wav'


# In[11]:


clean_data, samplerate = sf.read(filename) #Sound file returns float64 encoding


# In[12]:


#plotFile(filename)


# In[13]:


distorted = clean_data.copy()
height, width = distorted.shape
distorted += 0.005 * np.random.randn(height, width)
sf.write('distorted.wav',distorted,samplerate)
#plotFile('distorted.wav')


# In[14]:


noise = clean_data - distorted
sf.write('noise.wav',noise,samplerate)
#plotFile('noise.wav')


# In[15]:


#Extract patches from clean file
patch_size = (100,2) #both channels together
patches = extract_patches_2d(clean_data,patch_size)


# In[16]:


patches.shape


# In[17]:


stacked_patches = patches.reshape(patches.shape[0], -1)


# In[18]:


stacked_patches.shape


# In[19]:


#normalize patches so we can learn the dictionary
norm_data = stacked_patches
norm_data -= np.mean(norm_data, axis=0)
norm_data /= np.std(norm_data, axis=0)


# In[20]:


#Learn dictionary
dictionary = MiniBatchDictionaryLearning(n_components=100, alpha=1, batch_size=10, n_iter=500)
V = dictionary.fit(norm_data).components_ #this is the dictionary


# In[21]:


#pre-process noisy file
distorted_patches = extract_patches_2d(distorted, patch_size)
distorted_stacked_patches = distorted_patches.reshape(distorted_patches.shape[0],-1)

#center the data
intercept = np.mean(distorted_stacked_patches, axis=0)
distorted_stacked_patches -= intercept


# In[22]:


#find sparse code of distorted image given the dictionary
dictionary.set_params(transform_algorithm='omp',transform_n_nonzero_coefs=16)
code = dictionary.transform(distorted_stacked_patches) #this is sparse code


# In[23]:


#reconstruct the patches from the dot product of the sparse code and dictionary
recon_patches = np.dot(code, V)


# In[24]:


#uncenter the reconstructed data
recon_patches += intercept
#unstack the reconstructed patches
recon_patches = recon_patches.reshape(len(distorted_stacked_patches), *patch_size)


# In[25]:


#reconstruct audio from patches
denoised = reconstruct_from_patches_2d(recon_patches, (height, width))


# In[26]:


sf.write('denoised.wav',denoised,samplerate)
plotFile('denoised.wav')


# In[27]:


delta = distorted - denoised
sf.write('delta.wav', delta, samplerate)
plotFile('delta.wav')


# In[28]:


plotFile(filename) #original file Alarm01.wav


# In[29]:


plotFile('distorted.wav') #distorted w/ guassian noise


# In[30]:


plotFile('noise.wav') #just the difference btw original and distorted


# In[32]:


plotFile('denoised.wav') #after minibatch dict learning batch size=10, 100 components
#code found with omp w/ 16 atoms


# In[33]:


plotFile('delta.wav') #difference btw denoised and original


# In[ ]:


### Need to try some other parameters for this
"""
lasso_lars, transform_alpha {.001, .01, .1, 1} 1 by default
lasso_cd, transform_alpha {.001, .01, .1, 1}
threshold, transform_alpha {.001, .01, .1, 1}
lars, transform_n_nonzero_coefs {2, 20, 40} default is .1*n_featurs, 20 in the above case

n_features is determined by the number dimensions*channels, 100 dictionaries, 2 channels1
omp, transform_n_nonzero_coefs {2, 20, 40}

??What's omp tolerance parameter. according to sklearn it 
overrides n_nonzero_coefs?




# In[ ]:


transform_algorithms = [
    ('lasso_lars\n alpha=.001', 'lasso_lars', {'transform_alpha': 0.001}),
    ('lasso_lars\n alpha=.01', 'lasso_lars', {'transform_alpha': 0.01}),
    ('lasso_lars\n alpha=.1', 'lasso_lars', {'transform_alpha': 0.1}),
    ('lasso_lars\n alpha=1', 'lasso_lars', {'transform_alpha': 1}),
    ('lasso_cd\n alpha=.001', 'lasso_cd', {'transform_alpha': .001}),
    ('lasso_cd\n alpha=.01', 'lasso_cd', {'transform_alpha': .01}),
    ('lasso_cd\n alpha=.1', 'lasso_cd', {'transform_alpha': .1}),
    ('lasso_cd\n alpha=1', 'lasso_cd', {'transform_alpha': 1}),
    ('threshold\n alpha=.001', 'threshold', {'transform_alpha': .001}),
    ('threshold\n alpha=.01', 'threshold', {'transform_alpha': .01}),
    ('threshold\n alpha=.1', 'threshold', {'transform_alpha': .1}),
    ('threshold\n alpha=1', 'threshold', {'transform_alpha': 1}),
    ('lars\n2 atom', 'lars', {'transform_n_nonzero_coefs': 2}),
    ('lars\n20 atom', 'lars', {'transform_n_nonzero_coefs': 20}),
    ('lars\n40 atom', 'lars', {'transform_n_nonzero_coefs': 40})#,
    #need to do omp
]


# In[ ]:


#for loop on the transform_algorithms to compare all these against the task
#need a print function wo we can compare these


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


channel1 = data[:,0]
channel1


# In[6]:


channel2 = data[:,-1]
channel2 


# In[8]:


playsound(filename)


# In[11]:


playsound('distorted.wav')


# In[45]:


plotFile(filename)


# In[46]:


plotFile('distorted.wav')


# In[25]:


with wave.open(filename, 'r') as wav_file:
    signal = wav_file.readframes(-1)
    signal = np.frombuffer(signal, 'float32')
    
    channels = [[] for channel in range(wav_file.getnchannels())]
    for index, datum in enumerate(signal):
        channels[index%len(channels)].append(datum)
        
    fs = wav_file.getframerate()
    Time = np.linspace(0, len(signal)/len(channels)/fs, num=len(signal)/len(channels))
    
    plt.figure(1)
    plt.title('Signal Wave')
    for channel in channels:
        plt.plot(Time, channel)
    plt.show()


# In[14]:





# In[ ]:


Time=np.linspace(0, len(signal16i)/c1_signal.getframerate(),num=len(signal16i))
Time


# In[ ]:





# In[ ]:





# In[ ]:


plt.figure(1)
plt.title('Signal Wave')
plt.plot(Time,signal16i)
plt.show()


# In[ ]:





# In[8]:


fSample, samples = wavfile.read(filename)


# In[9]:


fSample


# In[10]:


samples


# In[11]:


samples.shape


# In[12]:


c1=samples[:,0]


# In[13]:


np.asarray(c1)


# In[14]:


c2=samples[:,-1]


# In[15]:


np.asarray(c2)


# In[16]:


wavfile.write('ringC1.wav',fSample, c1)
wavfile.write('ringC2.wav',fSample, c2)


# In[17]:


playsound('ringC1.wav')


# In[18]:


playsound('ringC2.wav')


# In[19]:


#c1_32fp = c1 / 32767
#c2_32fp = c2 /32767


# In[20]:


np.amin(c1)


# In[21]:


np.amax(c1)


# In[22]:


distorted = c1.copy()


# In[44]:


c1_signal = wave.open('ringC1.wav')


# In[45]:


c1_signal.getnchannels()


# In[46]:


c1_signal


# In[47]:


signal16i = c1_signal.readframes(-1)
signal16i = np.frombuffer(signal16i, dtype='int16')


# In[64]:


Time=np.linspace(0, len(signal16i)/c1_signal.getframerate(),num=len(signal16i))
Time


# In[65]:


plt.figure(1)
plt.title('Signal Wave')
plt.plot(Time,signal16i)
plt.show()


# In[71]:


#Now let's noise it up
distorted = samples.copy()


# In[73]:


height, width = distorted.shape


# In[75]:


distorted 


# In[74]:


distorted += 0.175*np.random.randn(height, width)


# In[ ]:


"""

