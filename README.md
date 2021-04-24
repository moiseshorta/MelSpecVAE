# MelSpecVAE
by Moisés Horta Valenzuela, 2021

Website: <a href=http://moiseshorta.audio>moiseshorta.audio</a>

Twitter: <a href=http://twitter.com/hexorcismos>@hexorcismos</a>

MelSpecVAE is a Variational Autoencoder that can synthesize Mel-Spectrograms which can be inverted into raw audio waveform.
Currently you can train it with any dataset of .wav audio at 44.1khz Sample Rate and 16bit bitdepth.
 
> Features:
* Interpolate through 2 different points in the latent space and synthesize the 'in between' sounds.
* Generate short one-shot audio
* Synthesize arbitrarily long audio samples by generating seeds and sample from the latent space. 
  Noise types for generating Z-vectors are uniform, Perlin and fractal.
 
> Credits:
* VAE neural network architecture coded following 'The Sound of AI' Youtube tutorial series by Valerio Velardo
* Some utility functions from Marco Passini's MelGAN-VC Jupyter Notebook.
