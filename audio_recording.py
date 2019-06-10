
# coding: utf-8

# In[3]:


import pyaudio
import wave        


# In[4]:


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 25600
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "recorded.wav"


# In[5]:


def recording(WAVE_OUTPUT_FILENAME):
    
    #오디오 객체생성
    p = pyaudio.PyAudio()
    #stream = pyaudio로 open 하는것
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("* recording")

    frames = []

    #읽은 데이터 있는동안 
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)     #데이터 wav파일에서 읽기
        frames.append(data)

    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    


# In[ ]:









