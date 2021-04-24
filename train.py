import tensorflow as tf
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from glob import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from numpy import linspace
import soundfile as sf
from utility import get_time_stamp

#Hyperparameters
LEARNING_RATE = 0.0005
EPOCHS =  40
BATCH_SIZE  = 64 
VECTOR_DIM = 64

hop=256               #hop size (window size = 4*hop)
sr=44100              #sampling rate
min_level_db=-100     #reference values to normalize data
ref_level_db=20

shape=128           #length of time axis of split specrograms
spec_split=1

#Waveform to Spectrogram conversion

''' Decorsière, Rémi, Peter L. Søndergaard, Ewen N. MacDonald, and Torsten Dau.
"Inversion of auditory spectrograms, traditional spectrograms, and other envelope representations."
IEEE/ACM Transactions on Audio, Speech, and Language Processing 23, no. 1 (2014): 46-56.'''

#ORIGINAL CODE FROM https://github.com/yoyololicon/spectrogram-inversion

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
import math
import heapq
from torchaudio.transforms import MelScale, Spectrogram

torch.set_default_tensor_type('torch.cuda.FloatTensor')

specobj = Spectrogram(n_fft=4*hop, win_length=4*hop, hop_length=hop, pad=0, power=2, normalized=False)
specfunc = specobj.forward
melobj = MelScale(n_mels=hop, sample_rate=sr, f_min=0.)
melfunc = melobj.forward

def melspecfunc(waveform):
  specgram = specfunc(waveform)
  mel_specgram = melfunc(specgram)
  return mel_specgram

def spectral_convergence(input, target):
    return 20 * ((input - target).norm().log10() - target.norm().log10())

def GRAD(spec, transform_fn, samples=None, init_x0=None, maxiter=1000, tol=1e-6, verbose=1, evaiter=10, lr=0.002):

    spec = torch.Tensor(spec)
    samples = (spec.shape[-1]*hop)-hop

    if init_x0 is None:
        init_x0 = spec.new_empty((1,samples)).normal_(std=1e-6)
    x = nn.Parameter(init_x0)
    T = spec

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam([x], lr=lr)

    bar_dict = {}
    metric_func = spectral_convergence
    bar_dict['spectral_convergence'] = 0
    metric = 'spectral_convergence'

    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            optimizer.zero_grad()
            V = transform_fn(x)
            loss = criterion(V, T)
            loss.backward()
            optimizer.step()
            lr = lr*0.9999
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr

            if i % evaiter == evaiter - 1:
                with torch.no_grad():
                    V = transform_fn(x)
                    bar_dict[metric] = metric_func(V, spec).item()
                    l2_loss = criterion(V, spec).item()
                    pbar.set_postfix(**bar_dict, loss=l2_loss)
                    pbar.update(evaiter)

    return x.detach().view(-1).cpu()

def normalize(S):
  return np.clip((((S - min_level_db) / -min_level_db)*2.)-1., -1, 1)

def denormalize(S):
  return (((np.clip(S, -1, 1)+1.)/2.) * -min_level_db) + min_level_db

def prep(wv,hop=192):
  S = np.array(torch.squeeze(melspecfunc(torch.Tensor(wv).view(1,-1))).detach().cpu())
  S = librosa.power_to_db(S)-ref_level_db
  return normalize(S)

def deprep(S):
  S = denormalize(S)+ref_level_db
  S = librosa.db_to_power(S)
  wv = GRAD(np.expand_dims(S,0), melspecfunc, maxiter=2500, evaiter=10, tol=1e-8)
  return np.array(np.squeeze(wv))

#---------Helper functions------------#

#Generate spectrograms from waveform array
def tospec(data):
  specs=np.empty(data.shape[0], dtype=object)
  for i in range(data.shape[0]):
    x = data[i]
    S=prep(x)
    S = np.array(S, dtype=np.float32)
    specs[i]=np.expand_dims(S, -1)
  print(specs.shape)
  return specs

#Generate multiple spectrograms with a determined length from single wav file
def tospeclong(path, length=4*44100):
  x, sr = librosa.load(path,sr=44100)
  x,_ = librosa.effects.trim(x)
  loudls = librosa.effects.split(x, top_db=50)
  xls = np.array([])
  for interv in loudls:
    xls = np.concatenate((xls,x[interv[0]:interv[1]]))
  x = xls
  num = x.shape[0]//length
  specs=np.empty(num, dtype=object)
  for i in range(num-1):
    a = x[i*length:(i+1)*length]
    S = prep(a)
    S = np.array(S, dtype=np.float32)
    try:
      sh = S.shape
      specs[i]=S
    except AttributeError:
      print('spectrogram failed')
  print(specs.shape)
  return specs

#Waveform array from path of folder containing wav files
def audio_array(path):
  ls = glob(f'{path}/*.wav')
  adata = []
  for i in range(len(ls)):
    x, sr = tf.audio.decode_wav(tf.io.read_file(ls[i]), 1)
    x = np.array(x, dtype=np.float32)
    adata.append(x)
  return np.array(adata)

#Concatenate spectrograms in array along the time axis
def testass(a):
  but=False
  con = np.array([])
  nim = a.shape[0]
  for i in range(nim):
    im = a[i]
    im = np.squeeze(im)
    if not but:
      con=im
      but=True
    else:
      con = np.concatenate((con,im), axis=1)
  return np.squeeze(con)

#Split spectrograms in chunks with equal size
def splitcut(data):
  ls = []
  mini = 0
  minifinal = spec_split*shape   #max spectrogram length
  for i in range(data.shape[0]-1):
    if data[i].shape[1]<=data[i+1].shape[1]:
      mini = data[i].shape[1]
    else:
      mini = data[i+1].shape[1]
    if mini>=3*shape and mini<minifinal:
      minifinal = mini
  for i in range(data.shape[0]):
    x = data[i]
    if x.shape[1]>=3*shape:
      for n in range(x.shape[1]//minifinal):
        ls.append(x[:,n*minifinal:n*minifinal+minifinal,:])
      ls.append(x[:,-minifinal:,:])
  return np.array(ls)


#Training functions

def train(x_train, learning_rate, batch_size, epochs):
  vae = VAE(
      input_shape = (hop, shape*spec_split, 1),
      conv_filters=(512, 256, 128, 64, 32),
      conv_kernels=(3, 3, 3, 3, 3),
      conv_strides=(2, 2, 2, 2, (2,1)),
      latent_space_dim = VECTOR_DIM
  )
  vae.summary()
  vae.compile(learning_rate)
  vae.train(x_train, batch_size, epochs)
  return vae

def train_tfdata(x_train, learning_rate, epochs=10):
  vae = VAE(
      input_shape = (hop, 3*shape, 1),
      conv_filters=(512, 256, 128, 64, 32),
      conv_kernels=(3, 3, 3, 3, 3),
      conv_strides=(2, 2, 2, 2, (2,1)),
      latent_space_dim = VECTOR_DIM
  )
  vae.summary()
  vae.compile(learning_rate)
  vae.train(x_train, num_epochs=epochs)
  return vae

def continue_training(checkpoint):
  vae = VAE.load(checkpoint)
  vae.summary()
  vae.compile(LEARNING_RATE)
  vae.train(adata,BATCH_SIZE,EPOCHS)
  return vae

def load_model(checkpoint):
  vae = VAE.load(checkpoint)
  vae.summary()
  vae.compile(LEARNING_RATE)
  return vae

"""## Training"""

#Import folder containing .wav files for training
#Generating Mel-Spectrogram dataset (Uncomment where needed)
#adata: source spectrograms

audio_directory = "/home/hexorcismos/Desktop/AI/MelGAN-VC/Datasets/aerofonos"

#AUDIO TO CONVERT
awv = audio_array(audio_directory)         #get waveform array from folder containing wav files
aspec = tospec(awv)                        #get spectrogram array
adata = splitcut(aspec)                    #split spectrogams to fixed
print(np.shape(adata))

#Start training from scratch or resume training

training_run_name = "aerofonos_test_train" 
checkpoint_save_directory = "/home/hexorcismos/Desktop/AI/MelSpecVAE/checkpoints/"
resume_training = False 
resume_training_checkpoint_path = "/content/drive/MyDrive/MelSpecVAE/Amazondotcom_e320_22_4_2021_1417_h256_w128_z64" 
current_time = get_time_stamp()

if __name__ == "__main__":
    from vae import VAE

    if not resume_training:
      vae = train(adata, LEARNING_RATE, BATCH_SIZE, EPOCHS)
      vae.save(f"{checkpoint_save_directory}{training_run_name}_{current_time}_h{hop}_w{shape}_z{VECTOR_DIM}")
    else:
      vae = continue_training(resume_training_checkpoint_path)
      vae.save(f"{checkpoint_save_directory}{training_run_name}_{current_time}_h{hop}_w{shape}_z{VECTOR_DIM}")
