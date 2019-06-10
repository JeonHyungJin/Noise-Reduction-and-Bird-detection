
# coding: utf-8

# In[1]:


import fastversion_rasp_test as F
import audio_recording as R
import numpy as np
import noise_reduction as nr
import os #운영체제에서 제공되는 여러 기능을 파이썬에서 이용할 수 있도록 하는 모듈
from os import listdir #디렉터리에 있는 파일들의 리스트를 구하기 위해
from os.path import isfile, join #파일인지 확인 및 디렉터리와 파일명을 이어주는 기능 
import csv
import pandas as pd
# In[2]:


#하이퍼파라미터
ROOT_PATH = os.getcwd()
PATH_TEST = ROOT_PATH + '/data/Test/wav'
audio_path = './'
file_name = 'filteredOutput.wav'
n_mels = 40
n_frame = 500
window_size=1024
hop_size=512
sample_rate= 44100
pred = 0
header = ['itemid','datasetid','hasbird']
dataset = 'ff1010bird'

# In[ ]:
# csv file 
pred_csv_file = open('ff1010bird_metadata_2018_pred.csv', 'a')
pred_csv_writer = csv.writer(pred_csv_file)
#pred_csv_writer.writerow(header)

#모델 준비
# 1. data preprocessing
detect_model, X, sess = nr.ready_model()

# 2. classification
model = F.Sequential()
model = F.build_model(model)
model = F.load_weight(model)
# compile은 한번만, 할 때마다 메모리 잡아 먹어서, 테스팅 시 느려짐.
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

predResult = []
filelist = [f for f in listdir(PATH_TEST) if f.endswith(".wav")]
totalLen = len(filelist)
startpoint = 700
while startpoint < totalLen:
    testFile = filelist[startpoint]
    #음성 데이터 전처리
    y, sr = nr.run_modeltest(detect_model,X, sess, sample_rate,testFile)

    #저장된 파일로 detection.
    S=F.melspectrogram(y=y, sr=sr,n_fft=window_size, hop_length=hop_size, power=2.0, n_mels=40)
    S=S[:,0:500]
    X_test = S
    X_test = np.reshape(X_test,(-1,40,500,1))

    #모델테스트
    result=F.run_model(model,X_test)
    if(result>0.5) : # need to chang threshold.
        pred=1
        
    else:
        pred=0

    testFileName = os.path.splitext(testFile)[0] 
    startpoint = startpoint+1
    print(startpoint)
    # CSV로 파일 쓰기
    predResult.append([testFileName, dataset, pred])
    if startpoint % 100 == 0:
        pred_csv_writer.writerows(predResult)
        predResult.clear()

pred_csv_writer.writerows(predResult)
pred_csv_file.close()
#%%
