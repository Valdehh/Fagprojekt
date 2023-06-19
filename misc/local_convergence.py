############################################################################################################
# local_convergence.py
# This script is to calculate the test ELBOs for the two models.
############################################################################################################



import numpy as np
import matplotlib.pyplot as plt



folder = 'semi/'

print('Semi-supervised VAE')

ELBOs =np.load(folder + "train_ELBOs.npz")['train_ELBOs']
KLs =    np.load(folder + "train_KLs.npz")['train_KLs']
REs =    np.load(folder + "train_REs.npz")['train_REs']


print(ELBOs[-1], np.mean(ELBOs[-100:]))
print(KLs[-1], np.mean(KLs[-100:]))
print(REs[-1], np.mean(REs[-100:]))


folder = 'final_vae2/'

print('Vanilla VAE')

ELBOs = np.load(folder + "train_ELBOs.npz")['train_ELBOs']
KLs =    np.load(folder + "train_KLs.npz")['train_KLs']
REs =    np.load(folder + "train_REs.npz")['train_REs']

print(ELBOs[-1], np.mean(ELBOs[-100:]))
print(KLs[-1], np.mean(KLs[-100:]))
print(REs[-1], np.mean(REs[-100:]))

