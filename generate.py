# -*- coding: utf-8 -*-

"""## Run the next cells first for training or generating"""

#import tensorflow as tf
from vae import VAE
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
import time

from perlin_numpy import (
    generate_fractal_noise_2d, generate_fractal_noise_3d,
    generate_perlin_noise_2d, generate_perlin_noise_3d
)

#@title Hyperparameters
learning_rate = 0.0005 #@param {type:"raw"}
num_epochs_to_train =  40#@param {type:"integer"}
batch_size = 64 #@param {type:"integer"}
vector_dimension = 64 #@param {type:"integer"}

hop=256               #hop size (window size = 4*hop)
sr=44100              #sampling rate
min_level_db=-100     #reference values to normalize data
ref_level_db=20

LEARNING_RATE = learning_rate
BATCH_SIZE = batch_size
EPOCHS = num_epochs_to_train
VECTOR_DIM=vector_dimension

shape=128           #length of time axis of split specrograms
spec_split=1

#@title Waveform to Spectrogram conversion

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

#@title Helper functions

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
def tospeclong(path, length=4*sr):
  x, sr = librosa.load(path,sr=sr)
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

"""## Generation"""

#@title Load Checkpoint for Generating

checkpoint_load_directory = "/home/hexorcismos/Desktop/AI/MelSpecVAE/checkpoints/Amazondotcom_e360_22_4_2021_151_h256_w128_z64" #@param {type:"string"}

#-------LOAD MODEL FOR GENERATING-------------#

vae = VAE.load(checkpoint_load_directory)
print("Loaded checkpoint")


#-----TESTING FUNCTIONS ----------- #

def select_spec(spec, labels, num_spec=10):
    sample_spec_index = np.random.choice(range(len(spec)), num_spec)
    sample_spec = spec[sample_spec_index]
    sample_labels = labels[sample_spec_index]
    return sample_spec, sample_labels


def plot_reconstructed_spec(spec, reconstructed_spec):
    fig = plt.figure(figsize=(15, 3))
    num_spec = len(spec)
    for i, (image, reconstructed_image) in enumerate(zip(spec, reconstructed_spec)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_spec, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_spec, i + num_spec + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()


def plot_spec_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()

#---------------NOISE GENERATOR FUNCTIONS ------------#

def generate_random_z_vect(seed=1001,size_z=1,scale=1.0):
    np.random.seed(seed)
    x = np.random.uniform(low=(scale * -1.0), high=scale, size=(size_z,VECTOR_DIM))
    return x

def generate_z_vect_from_perlin_noise(seed=1001, size_z=1, scale=1.0):
    np.random.seed(seed)
    x = generate_perlin_noise_2d((size_z, VECTOR_DIM), (1,1))
    x = x*scale
    return x

def generate_z_vect_from_fractal_noise(seed=1001, size_z=1, scale=1.0,):
    np.random.seed(seed)
    x = generate_fractal_noise_2d((size_z, VECTOR_DIM), (1,1),)
    x = x*scale
    return x


#-------SPECTROGRAM AND SOUND SYNTHESIS UTILITY FUNCTIONS -------- #

#Assembling generated Spectrogram chunks into final Spectrogram
def specass(a,spec):
  but=False
  con = np.array([])
  nim = a.shape[0]
  for i in range(nim-1):
    im = a[i]
    im = np.squeeze(im)
    if not but:
      con=im
      but=True
    else:
      con = np.concatenate((con,im), axis=1)
  diff = spec.shape[1]-(nim*shape)
  a = np.squeeze(a)
  con = np.concatenate((con,a[-1,:,-diff:]), axis=1)
  return np.squeeze(con)

#Splitting input spectrogram into different chunks to feed to the generator
def chopspec(spec):
  dsa=[]
  for i in range(spec.shape[1]//shape):
    im = spec[:,i*shape:i*shape+shape]
    im = np.reshape(im, (im.shape[0],im.shape[1],1))
    dsa.append(im)
  imlast = spec[:,-shape:]
  imlast = np.reshape(imlast, (imlast.shape[0],imlast.shape[1],1))
  dsa.append(imlast)
  return np.array(dsa, dtype=np.float32)

#Converting from source Spectrogram to target Spectrogram
def towave_reconstruct(spec, spec1, name, path='../content/', show=False, save=False):
  specarr = chopspec(spec)
  specarr1 = chopspec(spec1)
  print(specarr.shape)
  a = specarr
  print('Generating...')
  ab = specarr1
  print('Assembling and Converting...')
  a = specass(a,spec)
  ab = specass(ab,spec1)
  awv = deprep(a)
  abwv = deprep(ab)
  if save:
    print('Saving...')
    pathfin = f'{path}/{name}'
    sf.write(f'{pathfin}.wav', awv, sr)
    print('Saved WAV!')
  if show:
    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(np.flip(a, -2), cmap=None)
    axs[0].axis('off')
    axs[0].set_title('Reconstructed')
    axs[1].imshow(np.flip(ab, -2), cmap=None)
    axs[1].axis('off')
    axs[1].set_title('Input')
    plt.show()
  return abwv

#Converting from Z vector generated spectrogram to waveform
def towave_from_z(spec, name, path='../content/', show=False, save=False):
  specarr = chopspec(spec)
  print(specarr.shape)
  a = specarr
  print('Generating...')
  print('Assembling and Converting...')
  a = specass(a,spec)
  awv = deprep(a)
  if save:
    print('Saving...')
    pathfin = f'{path}/{name}'
    sf.write(f'{pathfin}.wav', awv, sr)
    print('Saved WAV!')
  if show:
    fig, axs = plt.subplots(ncols=1)
    axs.imshow(np.flip(a, -2), cmap=None)
    axs.axis('off')
    axs.set_title('Decoder Synthesis')
    plt.show()
  return awv

#Generate one-shot samples from latent space with random or manual seed
def one_shot_gen(num_samples=1, use_seed=False, seed=1001, z_scale=-2.2, save=True, name="one_shot", path="/home/hexorcismos/Desktop/AI/MelSpecVAE/results"):
    num_samples_to_generate =   num_samples
    _use_seed = use_seed
    _seed = seed
    scale_z_vectors = z_scale
    save_audio = save
    audio_name = name
    audio_save_directory = path

    y = np.random.randint(0, 2**32-1)  # generated random int to pass and convert into vector
    i=0
    while i < num_samples_to_generate:
      if not _use_seed:
        z = generate_random_z_vect(y, num_samples_to_generate,scale=scale_z_vectors)
      else:
        z = generate_random_z_vect(_seed, num_samples_to_generate,scale=scale_z_vectors)
      z_sample = np.array(vae.sample_from_latent_space(z))
      towave_from_z(z_sample[i], name=f'{audio_name}_{i}',path=audio_save_directory,show=False, save=save_audio)
      i+=1

    if not _use_seed:
      print("Generated from seed:", y)
    else:
      print("Generated from seed:", _seed)

#Generate arbitrary long audio from latent space with random or custom seed using uniform, Perlin or fractal noise
def noise_gen(num_samples=1, _noise_type="fractal", _use_seed=False, _seed=1001, z_scale=2.5, save=False, name="noise_generation", path="/home/hexorcismos/Desktop/AI/MelSpecVAE/results"):
    num_seeds_to_generate = num_samples
    noise_type = _noise_type #params are ["uniform", "perlin", "fractal"]
    use_seed = _use_seed
    seed = _seed
    scale_z_vectors =  z_scale
    save_audio = save
    audio_name = name
    audio_save_directory = path


    y = np.random.randint(0, 2**32-1)                         # generated random int to pass and convert into vector
    if not use_seed:
      if noise_type == "uniform":
        z = generate_random_z_vect(y, num_seeds_to_generate,scale_z_vectors)            # vectors to input into latent space
      if noise_type == "perlin":
        z = generate_z_vect_from_perlin_noise(y, num_seeds_to_generate,scale_z_vectors)            # vectors to input into latent space
      if noise_type == "fractal":
        z = generate_z_vect_from_fractal_noise(y, num_seeds_to_generate,scale_z_vectors)            # vectors to input into latent space
    if use_seed:
      if noise_type == "uniform":
        z = generate_random_z_vect(seed, num_seeds_to_generate,scale_z_vectors)            # vectors to input into latent space
      if noise_type == "perlin":
        z = generate_z_vect_from_perlin_noise(seed, num_seeds_to_generate,scale_z_vectors)            # vectors to input into latent space
      if noise_type == "fractal":
        z = generate_z_vect_from_fractal_noise(seed, num_seeds_to_generate,scale_z_vectors)            # vectors to input into latent space
    z_sample = np.array(vae.sample_from_latent_space(z))
    assembled_spec = testass(z_sample)
    towave_from_z(assembled_spec,audio_name,audio_save_directory,show=False,save=save_audio)

    if not use_seed:
      print("Generated from seed:", y)
    else:
      print("Generated from seed:", seed)

#Interpolate between two seeds for n-amount of steps
def interp_gen(num_samples=1, _use_seed=False, _seed=1001, interp_steps=5, z_scale=-2.2, interp_scale=1.2, save=False, name="one_shot", path="/content/"):
    use_seed = _use_seed #@param {type:"boolean"}
    seed =  _seed #@param {type:"slider", min:0, max:4294967295, step:1}
    num_interpolation_steps = interp_steps#@param {type:"integer"}
    scale_z_vectors =  z_scale #@param {type:"slider", min:-5.0, max:5.0, step:0.1}
    scale_interpolation_ratio =  interp_scale #@param {type:"slider", min:-5.0, max:5.0, step:0.1}
    save_audio = save #@param {type:"boolean"}
    audio_name = name #@param {type:"string"}
    audio_save_directory = path #@param {type:"string"}

    # generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, n_samples, n_classes=10):
    	# generate points in the latent space
    	x_input = randn(latent_dim * n_samples)
    	# reshape into a batch of inputs for the network
    	z_input = x_input.reshape(n_samples, latent_dim)
    	return z_input

    # uniform interpolation between two points in latent space
    def interpolate_points(p1, p2,scale, n_steps=10):
    	# interpolate ratios between the points
    	ratios = linspace(-scale, scale, num=n_steps)
    	# linear interpolate vectors
    	vectors = list()
    	for ratio in ratios:
    		v = (1.0 - ratio) * p1 + ratio * p2
    		vectors.append(v)
    	return asarray(vectors)

    y = np.random.randint(0, 2**32-1)
    if not use_seed:
      pts = generate_random_z_vect(y,num_samples,scale_z_vectors)
    else:
      pts = generate_random_z_vect(seed,num_samples,scale_z_vectors)

    # interpolate points in latent space
    interpolated = interpolate_points(pts[0], pts[1], scale_interpolation_ratio, num_interpolation_steps)
    #print(np.shape(interpolated))
    interp = np.array(vae.sample_from_latent_space(interpolated))
    assembled_spec = testass(interp)
    towave_from_z(assembled_spec,audio_name,audio_save_directory,show=False, save=save_audio)
    #print(np.shape(assembled_spec))

    if not use_seed:
      print("Generated from seed:", y)
    else:
      print("Generated from seed:", seed)

if __name__ == "__main__":
    #one_shot_gen(num_samples=10, name="amazondotcom_test")
    #noise_gen(num_samples=300,_use_seed=False,_noise_type="fractal", z_scale=2.5, name="uniform_test2s", save=True)
    interp_gen(num_samples=10, _use_seed=False, _seed=1001, interp_steps=30, z_scale=-1.5, interp_scale=1.0, save=True, name="interp_test2", path="/home/hexorcismos/Desktop/AI/MelSpecVAE/results")
