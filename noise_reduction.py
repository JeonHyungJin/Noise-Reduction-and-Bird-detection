
# coding: utf-8 
# # Noise reduction
# 
#  1. clean sound + noise sound = new sound
#  2. design DNN model - X : new sound, Y : clean sound
#  3. test #  

# ## 1. make new sound combined clean sound and noise sound

# import modules
import os #운영체제에서 제공되는 여러 기능을 파이썬에서 이용할 수 있도록 하는 모듈
import librosa #음악과 오디오 분석과 관련된 기능을 제공하는 모듈
import librosa.display
from os import listdir #디렉터리에 있는 파일들의 리스트를 구하기 위해
from os.path import isfile, join #파일인지 확인 및 디렉터리와 파일명을 이어주는 기능 
import numpy as np #수학 및 과학 연산을 위한 모듈
import scipy.io.wavfile #수학 및 과학 연산을 위한 scipy 모듈 중 특별히 wavfile IO와 관련된 모듈
import pickle #객체의 형태를 그대로 유지하면서 파일을 저장하고 불러올 수 있게 하는 모듈
import tensorflow as tf
import pickle
import random
import wave
import array
import matplotlib.pyplot as plt

# declare path
ROOT_PATH = os.getcwd() + "/.."
PATH_NOISE = ROOT_PATH + '/data/noise'
PATH_CLEAN = ROOT_PATH + '/data/clean'
PATH_FEATURE_OUTPUT = ROOT_PATH + '/output/feature'
PATH_TEST = ROOT_PATH + '/data/Test/wav'
PATH_RECORD = ROOT_PATH +'/record'
FILE_EXCEPT = ROOT_PATH +'err.txt'
FFT_SIZE = 512

# In[133]:
## 음원 데이터 정규화 작업 ##
def rms(input):
    rms = np.sqrt(np.mean(input**2)) 
    return rms

def equalizingRMS(source,target):
    target = rms(source)/rms(target)*target
    return target
########################

## 순수 음원 + 소음 작업 : snr 비율로 ##
def addNoise(speech,noise,snr):
    if len(speech) > len(noise):
        speech = speech[0:len(noise)]
    else:
        noise = noise[0:len(speech)]
        
    noisy = speech + np.sqrt(np.sum(np.abs(speech)**2))/ np.sqrt(np.sum(np.abs(noise)**2) * np.power(10,snr*0.1)) * noise
    return noisy
####################################

## 모델 학습에 사용될 input, output 데이터 생성 ##
## stft 이용해 특징 추출하여 pickle 형식으로 저장 ##
def soundFeature():
    f = open(FILE_EXCEPT,'w')
    standard_wav, rate = librosa.load(PATH_CLEAN + '/bird1.wav', sr=None)
    noisy_LPSs = []
    clean_LPSs = []
    
    for cleanFile in [f for f in listdir(PATH_CLEAN) if f.endswith(".wav")]:
        try :
            print(cleanFile)
            clean_wav, rate = librosa.load(PATH_CLEAN + '/' + cleanFile, sr=None)
            clean_wav = clean_wav[0:0 + len(standard_wav)]
            clean_wav = equalizingRMS(standard_wav, clean_wav)
            clean_LPS = librosa.stft(clean_wav, n_fft=FFT_SIZE)

        except Exception as e :
            print("Except :", cleanFile)
            f.write(cleanFile+"\n")
        else :
            for noiseFile in [f for f in listdir(PATH_NOISE) if f.endswith(".wav")]:
                noise_wav, rate = librosa.load(PATH_NOISE + '/' + noiseFile,sr=None)
                noise_wav = equalizingRMS(standard_wav, noise_wav)
                start=0
                split_noise_wav = noise_wav[start: start + len(clean_wav)]    

                noisy_wav = addNoise(clean_wav, split_noise_wav, -5)
                noisy_LPS = librosa.stft(noisy_wav,n_fft=FFT_SIZE)
                noisy_LPSs.append(noisy_LPS)
                clean_LPSs.append(clean_LPS)
    f.close()
    noisy_LPS_ = np.hstack(noisy_LPSs)
    clean_LPS_ = np.hstack(clean_LPSs)

    data = {'train_noisy' : noisy_LPS_, 'train_clean' : clean_LPS_}

    fileName = PATH_FEATURE_OUTPUT + "/train_feature.pickle"
    with open(fileName, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
##########################################

## model 학습 시 batch 단위로 나누어 데이터 정리 ##
def batch_creator(batch_size, X_train, y_train):
    """Create batch with random samples and return appropriate format"""
    X_train = X_train.T
    y_train = y_train.T
    
    dataset_length = X_train.shape[0]
    
    idx_batch = np.arange(dataset_length)
    #np.random.shuffle(idx_batch)
    total_batch = int(np.ceil(dataset_length/batch_size))
    
    batch_x = [X_train[idx_batch[i*batch_size:(i+1)*batch_size]] for i in range(total_batch)]
    batch_y = [y_train[idx_batch[i*batch_size:(i+1)*batch_size]] for i in range(total_batch)]

    return batch_x, batch_y
############################################

## 모델 학습 - regression ##
## epoch - 800, batch - 100
## 모델 저장 
def dnn():
    with open(PATH_FEATURE_OUTPUT + "/train_feature.pickle",'rb') as handle:
        data = pickle.load(handle)
    neuron = 1024
    fft_size = FFT_SIZE
    training_epochs = 800
    learning_rate = 1e-5
    batch_size = 100
    total_batch = int(data['train_noisy'].shape[1] / batch_size)
    
    tf.reset_default_graph()
    
    noisy_mean = np.mean(data['train_noisy'])
    noisy_std = np.std(data['train_noisy'])
    clean_mean = np.mean(data['train_clean'])
    clean_std = np.std(data['train_clean'])
    
    data['train_noisy'] = (data['train_noisy'] - noisy_mean) / (noisy_std)
    data['train_clean'] = (data['train_clean'] - clean_mean) / (clean_std)
    
    X = tf.placeholder(tf.float32, [None, (fft_size / 2 + 1)])
    Y = tf.placeholder(tf.float32, [None, (fft_size / 2 + 1)])

    W1 = tf.get_variable("W1", shape=[int(fft_size / 2 + 1), neuron], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.Variable([neuron]), name='bias1')
    L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), tf.cast(b1, tf.float32)))
    W2 = tf.get_variable("W2", shape=[neuron, neuron], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.Variable([neuron]), name='bias2')
    L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), tf.cast(b2, tf.float32)))
    W3 = tf.get_variable("W3", shape=[neuron, neuron], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.Variable([neuron]), name='bias3')
    L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), tf.cast(b3, tf.float32)))
    W4 = tf.get_variable("W4", shape=[neuron, neuron/2], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.Variable([neuron/2]), name='bias4')
    L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), tf.cast(b4, tf.float32)))
    W5 = tf.get_variable("W5", shape=[neuron/2, int(fft_size / 2 + 1)], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.Variable([neuron]), name='bias5')
    hypothesis =tf.add(tf.matmul(L4, W5), tf.cast(b5, tf.float32))
    
    # costfunction = mse, optimizer = Adam
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Save parameter
    param_list = [W1, W2, W3, W4,W5, b1, b2, b3, b4,b5]
    saver = tf.train.Saver(param_list)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        saver.save(sess, './model.ckpt')
        cur_batch = 0
        batch_x, batch_y = batch_creator(batch_size, data['train_noisy'], data['train_clean'])

        for i in range(total_batch):
            _, c = sess.run([optimizer,cost], feed_dict = {X: batch_x[i], Y: batch_y[i]})
            print("Epoch: ", '%02d' % (epoch+1)," Batch: ",'%02d'%i,"Cost=","{:.5f}".format(c))
################################################

## 테스트 할 파일의 특징 추출 ##
def test_feature_extraction(testFile='bird31.wav', noiseFile='traffic1.wav', snr=0):
    # =============================================================================
    # Read Noise Files in PATH_NOISE Folder (with normalized to standard waveform)
    # =============================================================================

    standard_wav, rate = librosa.load(PATH_CLEAN + '/bird1.wav', sr=None)

    noise_wav, rate = librosa.load(PATH_NOISE + '/' + noiseFile, sr=None) 
    noise_wav = equalizingRMS(standard_wav, noise_wav) 
    noise_lps = np.real(librosa.stft(noise_wav, n_fft=FFT_SIZE))
    # =============================================================================
    # Test Data Augmentation (with normalized to standard waveform)
    # =============================================================================

    test_wav, rate = librosa.load(PATH_TEST + '/' + testFile, sr=None)
    test_wav = test_wav[0:0 + len(test_wav)]
    test_wav = equalizingRMS(standard_wav, test_wav)

    plttest_lps = np.real(librosa.stft(test_wav, n_fft=FFT_SIZE))
    plt.subplot(311)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(plttest_lps), ref=np.max), y_axis='log', x_axis='time')    
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
	
    test_wav = addNoise(test_wav, noise_wav, snr)
    
    test_LPS = np.real(librosa.stft(test_wav,n_fft=FFT_SIZE))

    plt.subplot(312)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(test_LPS), ref=np.max), y_axis='log', x_axis='time')    
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    librosa.output.write_wav('./temptest.wav', test_wav, rate)
    return test_LPS, rate
##############################

## 테스트 작업 - 저장한 모델 읽어드리고, 특징 추출해서 모델에 input으로, output을 다시 음원으로 하는 작업 ##
def test():
    neuron = 1024
    fft_size = FFT_SIZE
    
    
    tf.reset_default_graph()
    
    # Add ops to save and restore all the variables.
    X = tf.placeholder(tf.float32, [None, (fft_size / 2 + 1)])

    # Later, launch the model, use the saver to restore variables from disk, and
    # do some work with the model.
    # run the session
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))

        # restore the saved vairable
        print("Model restored.")
        graph = tf.get_default_graph()
        W1 = graph.get_tensor_by_name("W1:0")
        W2 = graph.get_tensor_by_name("W2:0")
        W3 = graph.get_tensor_by_name("W3:0")
        W4 = graph.get_tensor_by_name("W4:0")
        W5 = graph.get_tensor_by_name("W5:0")
        b1 = graph.get_tensor_by_name("bias1:0")
        b2 = graph.get_tensor_by_name("bias2:0")
        b3 = graph.get_tensor_by_name("bias3:0")
        b4 = graph.get_tensor_by_name("bias4:0")
        b5 = graph.get_tensor_by_name("bias5:0")

        L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), tf.cast(b1, tf.float32)))
        L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), tf.cast(b2, tf.float32)))
        L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), tf.cast(b3, tf.float32)))
        L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), tf.cast(b4, tf.float32)))
        hypothesis = tf.add(tf.matmul(L4, W5), tf.cast(b5, tf.float32))
        # Check the values of the variables
        test, test_rate = test_feature_extraction(testFile='birdTest.wav' ,noiseFile='traffic1.wav', snr=-5)
        
        # trans~
        test = test.T

        # zero-mean and unit-variance normalization
        output_lps = sess.run([hypothesis], feed_dict = {X: test})[0]

        plt.subplot(313)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(output_lps.T), ref=np.max), y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig('./test_lps.jpg')
        
        sound_enhanced = librosa.istft(output_lps.T)
        librosa.output.write_wav('./testOutput.wav', sound_enhanced, test_rate)
####################################################

## 실제 사용할 때, 모델을 먼저 읽어 그래프 셋팅 ##
def ready_model():   
    fft_size = FFT_SIZE
    tf.reset_default_graph()
    
    # Add ops to save and restore all the variables.
    X = tf.placeholder(tf.float32, [None, (fft_size / 2 + 1)])

    sess = tf.Session()
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))

    # restore the saved vairable
    print("Model restored.")
    graph = tf.get_default_graph()
    W1 = graph.get_tensor_by_name("W1:0")
    W2 = graph.get_tensor_by_name("W2:0")
    W3 = graph.get_tensor_by_name("W3:0")
    W4 = graph.get_tensor_by_name("W4:0")
    W5 = graph.get_tensor_by_name("W5:0")
    b1 = graph.get_tensor_by_name("bias1:0")
    b2 = graph.get_tensor_by_name("bias2:0")
    b3 = graph.get_tensor_by_name("bias3:0")
    b4 = graph.get_tensor_by_name("bias4:0")
    b5 = graph.get_tensor_by_name("bias5:0")


    L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), tf.cast(b1, tf.float32)))
    L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), tf.cast(b2, tf.float32)))
    L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), tf.cast(b3, tf.float32)))
    L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), tf.cast(b4, tf.float32)))
    hypothesis = tf.add(tf.matmul(L4, W5), tf.cast(b5, tf.float32))

    return hypothesis, X, sess
######################################

## 모델 실행 ##
## 만들어진 소리는 filteredOutput.wav
## 작업 전과 후 음원 데이터의 스펙트로그램은 result_lps.jpg 
def run_model(model,X, sess, sample_rate,testFile='recorded.wav'):
    neuron = 1024
    fft_size = FFT_SIZE

    # equlizing RMS
    standard_wav, rate = librosa.load(PATH_CLEAN + '/bird1.wav', sr=None)
    test_wav, rate = librosa.load(PATH_RECORD + '/' + testFile, sr=None)
    test_wav = equalizingRMS(standard_wav, test_wav)

    #Do stft to recorded wav
    stft_recorded = np.real(librosa.stft(test_wav,n_fft=FFT_SIZE))

    # express the spectrogram of recorded wav
    plt.subplot(211)
    plt.title('Power spectrogram')
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(stft_recorded), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    #run model
    output_lps = sess.run([model], feed_dict = {X: stft_recorded.T})[0]

    plt.subplot(212)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(output_lps.T), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig('./result_lps.jpg')

    sound_enhanced = librosa.istft(output_lps.T)
    return sound_enhanced, sample_rate
######################################

## 테스트를 위한 함수 ##
## wav 파일 생성 x, 바로 다음에 오는 detection 모델의 input으로 연결
def run_modeltest(model,X, sess, sample_rate, testFile='recorded.wav'):
    neuron = 1024
    fft_size = FFT_SIZE

    # equlizing RMS
    standard_wav, rate = librosa.load(PATH_CLEAN + '/bird1.wav',sr=None)
    test_wav, rate = librosa.load(PATH_TEST + '/' + testFile,sr=None)
    test_wav = equalizingRMS(standard_wav, test_wav)

    #Do stft to recorded wav
    stft_recorded = np.real(librosa.stft(test_wav,n_fft=FFT_SIZE))

    #run model
    output_lps = sess.run([model], feed_dict = {X: stft_recorded.T})[0]
    
    sound_enhanced = librosa.istft(output_lps.T)
    return sound_enhanced, rate
####################

def main():
    print("----------------------------------")
    print("1. Making sound-feature file")
    print("----------------------------------")
    soundFeature()

    print("----------------------------------")
    print("2. Learning noise-reduction model")
    print("----------------------------------")
    dnn()

    print("----------------------------------")
    print("3. Testing new file")
    print("----------------------------------")
    # test()

main()


#%%
