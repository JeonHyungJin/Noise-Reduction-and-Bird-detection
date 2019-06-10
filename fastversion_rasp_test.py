
# coding: utf-8

# In[2]:


'''
평가방법 : positive를 높이는방식

본논문
- STFT magnitude Spectrun
- n=40 log mel filter bank

다른논문
-STFT maginitude spectogram
- n=80 mel scaled filter bank
- scale log magnitude
- batch nomalization (0,1)
- subtract mean overtime on spectogram (for remove frequency dependency noise = colored noise)

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
'''


# In[1]:

'''
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''

# In[13]:

#path라이브러리
import os

# Scientific Math 라이브러리  
import numpy as np
import six
import scipy.signal
import scipy.fftpack as fft
from numpy.lib.stride_tricks import as_strided

#audio read 라이브러리
import audioread
import resampy

#keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, GRU,Dropout, Flatten,Reshape,BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.vis_utils import plot_model


# In[ ]:

MODEL_SAVE_FOLDER_PATH = './model/'
audio_path = './'
n_mels = 40
n_frame = 500
window_size=1024
hop_size=512
sample_rate=25600


# In[15]:


#librosa.feature.mel_spectogram

def hz_to_mel(frequencies, htk=False):

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels

def mel_to_hz(mels, htk=False):
    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


# In[16]:


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)

def fft_frequencies(sr=22050, n_fft=2048):
 
    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)

def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
        norm=1):


    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn('Empty filters detected in mel frequency basis. '
                      'Some channels will produce empty responses. '
                      'Try increasing your sampling rate (and fmax) or '
                      'reducing n_mels.')

    return weights


# In[17]:


def get_window(window, Nx, fftbins=True):
    if six.callable(window):
        return window(Nx)

    elif (isinstance(window, (six.string_types, tuple)) or
          np.isscalar(window)):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ParameterError('Window size mismatch: '
                             '{:d} != {:d}'.format(len(window), Nx))
    else:
        raise ParameterError('Invalid window specification: {}'.format(window))

        
def pad_center(data, size, axis=-1, **kwargs):
        
    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be '
                              'at least input size ({:d})').format(size, n))

    return np.pad(data, lengths, **kwargs)

def valid_audio(y, mono=True):

    if not isinstance(y, np.ndarray):
        raise ParameterError('data must be of type numpy.ndarray')

    if not np.issubdtype(y.dtype, np.floating):
        raise ParameterError('data must be floating-point')

    if mono and y.ndim != 1:
        raise ParameterError('Invalid shape for monophonic audio: '
                             'ndim={:d}, shape={}'.format(y.ndim, y.shape))

    elif y.ndim > 2 or y.ndim == 0:
        raise ParameterError('Audio must have shape (samples,) or (channels, samples). '
                             'Received shape={}'.format(y.shape))

    if not np.isfinite(y).all():
        raise ParameterError('Audio buffer is not finite everywhere')

    return True


def frame(y, frame_length=2048, hop_length=512):   
    
    if not isinstance(y, np.ndarray):
        raise ParameterError('Input must be of type numpy.ndarray, '
                             'given type(y)={}'.format(type(y)))

    if y.ndim != 1:
        raise ParameterError('Input must be one-dimensional, '
                             'given y.ndim={}'.format(y.ndim))

    if len(y) < frame_length:
        raise ParameterError('Buffer is too short (n={:d})'
                             ' for frame_length={:d}'.format(len(y), frame_length))

    if hop_length < 1:
        raise ParameterError('Invalid hop_length: {:d}'.format(hop_length))

    if not y.flags['C_CONTIGUOUS']:
        raise ParameterError('Input buffer must be contiguous.')

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames


def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):
        # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Check audio is valid
    valid_audio(y)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    MAX_MEM_BLOCK = 2**8 * 2**10
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]]

    return stft_matrix


def _spectrogram(y=None, S=None, n_fft=2048, hop_length=512, power=1):
    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))**power
    return S, n_fft

def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   power=2.0, **kwargs):
 
    S, n_fft = _spectrogram(y=y, S=S, n_fft=n_fft, hop_length=hop_length, power=power)

    mel_basis = mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)


# In[18]:


'''
Do with scipy STFT
def _spectrogram2(y=None, S=None, n_fft=2048, hop_length=512, power=1):
    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        a,b,S = np.abs(scipy.signal.stft(y, nperseg=n_fft, noverlap=hop_length))**power
    return S, n_fft

def melspectrogram2_scipy(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   power=2.0, **kwargs):
 
    S, n_fft = _spectrogram2(y=y, S=S, n_fft=n_fft, hop_length=hop_length, power=power)

    mel_basis = mel(sr, n_fft, **kwargs)

    return np.dot(mel_basis, S)

filename = audio_path + 'test.wav'
y,sr=librosa.core.load(filename,sr=sample_rate)
S=melspectrogram2(y=y, sr=sr,n_fft=window_size, hop_length=hop_size, power=2.0, n_mels=40)

plt.figure(figsize=(10,4))
librosa.display.specshow(librosa.power_to_db(S,ref=np.max),y_axis='mel',x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectogram')
plt.tight_layout()


#load wav file with scipy
def load_wav_file(fname, smprate=16000):

    smprate_real, data = wavfile.read(fname)
    if smprate_real == smprate:
        data = data.astype(float)
    elif (smprate_real % smprate) == 0:
        # integer factor downsample
        smpfactor = smprate_real // smprate
        data = np.pad(
            data, [(0, (-len(data)) % smpfactor)], mode='constant')
        data = np.reshape(data, [len(data)//smpfactor, smpfactor])
        data = np.mean(data.astype(float), axis=1)
    else:
        newlen = int(np.ceil(len(data) * (smprate / smprate_real)))
        # FIXME this resample is very slow on prime length
        data = scipy.signal.resample(data, newlen).astype(float)
    return data 

y=load_wav_file(fname=filename,smprate=sample_rate)
'''


# In[19]:


#librosa.core.read

def buf_to_float(x, n_bytes=2, dtype=np.float32):

    # Invert the scale of the data
    scale = 1./float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)
def fix_length(data, size, axis=-1, **kwargs):
    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[slices]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data
def resample(y, orig_sr, target_sr, res_type='kaiser_best', fix=True, scale=False, **kwargs):

    # First, validate the audio buffer
    valid_audio(y, mono=False)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[-1] * ratio))

    if res_type == 'scipy':
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)
    else:
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.ascontiguousarray(y_hat, dtype=y.dtype)




def load(path, sr=22050, mono=True, offset=0.0, duration=None,
         dtype=np.float32, res_type='kaiser_best'):

    y = []
    with audioread.audio_open(os.path.realpath(path)) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))
                               * n_channels)

        n = 0

        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[:s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)

        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
            if mono:
                y = to_mono(y)

        if sr is not None:
            y = resample(y, sr_native, sr, res_type=res_type)

        else:
            sr = sr_native

    # Final cleanup for dtype and contiguity
    y = np.ascontiguousarray(y, dtype=dtype)

    return (y, sr)

    


# In[21]:


def load_weight(model):
    model.load_weights(MODEL_SAVE_FOLDER_PATH + 'bird_sound-' + '17-0.3943.hdf5')
    return model 

def run_model(model,X_test):
    Y_pred = model.predict(X_test)
    #print(Y_pred)
    return Y_pred 


# In[22]:


def build_model(layers):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(5, 5), input_shape=(40, 500,1), padding='same',activation='relu')) #어쩌면 40,500만해야할지두
    print(model.output_shape)
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(5,1)))
    model.add(Dropout(0.25))
    print(model.output_shape)

    model.add(Conv2D(96, (5, 5), padding='same',activation='relu'))
    print(model.output_shape)
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(Dropout(0.25))
    print(model.output_shape)

    model.add(Conv2D(96, (5, 5), padding='same',activation='relu'))
    print(model.output_shape)
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(Dropout(0.25))
    print(model.output_shape)

    model.add(Conv2D(96, (5, 5), padding='same', activation='relu'))
    print(model.output_shape)
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,1)))
    model.add(Dropout(0.25))
    print(model.output_shape)

    model.add(Reshape((96,500))) #문제될거같은데..
    print(model.output_shape)

    model.add(GRU(units=500, return_sequences=True))
    print(model.output_shape)

    model.add(GRU(units=500, return_sequences=True))
    print(model.output_shape)

    model.add(Reshape((96,500,1))) #문제될거같은데..2
    print(model.output_shape)

    model.add(MaxPooling2D(pool_size=(1,500)))
    print(model.output_shape)

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model


# In[25]:
'''
filename = audio_path + 'test.wav'
y,sr=load(filename,sr=sample_rate)

S=melspectrogram(y=y, sr=sr,n_fft=window_size, hop_length=hop_size, power=2.0, n_mels=40)
S=S[:,0:500]
X_test = S
X_test = np.reshape(X_test,(-1,40,500,1))
np.shape(X_test)


# In[26]:

model = Sequential()
model = build_model(model)
model = load_weight(model)
model = run_model(model)
'''


# In[24]:


#STFT

#Spectogram

#mel scale filter bank

#log magnitude

# batch nomalixation

# subtract mean over time on spectogram
#(X_train, Y_train), (X_validation, Y_validation) = mnist.load_data()

#X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
#X_validation = X_validation.reshape(X_validation.shape[0], 28, 28, 1).astype('float32') / 255

#Y_train = np_utils.to_categorical(Y_train, 10)
#Y_validation = np_utils.to_categorical(Y_validation, 10)

