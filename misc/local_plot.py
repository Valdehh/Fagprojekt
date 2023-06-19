############################################################################################################
# local_plot.py
# This script is to plot some of the training data (used in results section).
############################################################################################################



import numpy as np
import matplotlib.pyplot as plt

folder = 'semi/'

ELBOs =np.load(folder + "train_ELBOs.npz")['train_ELBOs']
KLs =    np.load(folder + "train_KLs.npz")['train_KLs']
REs =    np.load(folder + "train_REs.npz")['train_REs']


fig = plt.figure(figsize=(10, 5))
plt.plot(KLs, label='KL', alpha=0.6, linestyle='-', color='green', linewidth=2)
plt.plot(REs, label='RE', alpha=0.6, linestyle='-', color='red', linewidth=2)
plt.plot(ELBOs, label='ELBO', alpha=0.8, linestyle='-', color='black', linewidth=2)
plt.xticks(np.arange(0, 110000, 10000), [str(i) for i in np.arange(0, 110, 10)])
plt.xlabel('Epochs', size=20, fontweight='bold')
plt.title('ELBO-components', size=22, fontweight='bold')
plt.ylabel('Loss', size=20, fontweight='bold')
plt.legend(fontsize=20, loc='upper right')
plt.savefig('plots/semi_' + 'ELBO-components.png')
plt.show()
