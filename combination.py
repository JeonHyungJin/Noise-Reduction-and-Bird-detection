
# coding: utf-8

# In[1]:


import fastversion_rasp_test as F
import audio_recording as R
import numpy as np
import noise_reduction as nr


# In[2]:


#하이퍼파라미터
audio_path = './'
recorded_file = 'recorded.wav'
file_name = 'filteredOutput.wav'
file_name2 = 'temptest.wav'
n_mels = 40
n_frame = 500
window_size=1024
hop_size=512
sample_rate= 44100

# In[ ]:


#모델 준비
detect_model, X, sess = nr.ready_model()

model = F.Sequential()
model = F.build_model(model)
model = F.load_weight(model)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

i = 0
while(1) : 
    #음성 레코딩
    R.recording(audio_path+'record/'+recorded_file)

    #음성 데이터 전처리
    y, sr = nr.run_model(detect_model,X, sess, sample_rate,recorded_file)

    #detection.
    S=F.melspectrogram(y=y, sr=sr,n_fft=window_size, hop_length=hop_size, power=2.0, n_mels=40)
    S=S[:,0:500]
    X_test = S
    X_test = np.reshape(X_test,(-1,40,500,1))

    #모델테스트
    result=F.run_model(model,X_test)
    if(result>0.5) :
        print('result:',result,'True')
    else:
        print('result:',result,'False')