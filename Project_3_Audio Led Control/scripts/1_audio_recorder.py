import sounddevice as sd
import numpy as np
import scipy
from scipy.io.wavfile import write
import time

i = 0

for i in range(50):
    category = "off"
    filename = 'off_' + str(i)
    duration = 1  # seconds
    fs = 22050  # samples per second
    print("Speak now ->" , category)
    audio_rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    int_audio = (np.clip(audio_rec , -32767 , 32767)) * 32767
    write('data/' + category +"/"+ filename + ".wav" , fs ,int_audio.astype(np.int16))


    print("Recorded ->" , i)








