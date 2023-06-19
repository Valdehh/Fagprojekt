#############################################################################################################
# tweak_latent_unsup.py
# This script is used to tweak the latent space of the unsupervised VAE and plot the corresponding images.
#############################################################################################################

import numpy as np
import torch
import sys
sys.path.insert(1, 'C:/Users/andre/OneDrive/Skrivebord/Fagprojekt')
from final_vae import decoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

model = torch.load('C:/Users/andre/OneDrive/Skrivebord/Fagprojekt/decoder.pt', map_location=device)

latent = np.load('latent_space_vanilla.npz')['z']

for j in range(300):
    val=np.linspace(np.min(latent[:,j]), np.max(latent[:,j]), num=16)
    latent_tensor = torch.tensor(latent[18], device=device, dtype=torch.float)
    latent_tensor[0]=val[0]

    print(np.max(latent[:,j]))
    print(np.min(latent[:,j]))
    print(latent_tensor[j])
    
    fig, ax = plt.subplots(4, 4)
    ax = ax.flatten()
    for i in range(10):
        latent_tensor[j] = val[i]
        print(latent_tensor[j])
        mu, log_var = torch.split(model.forward(latent_tensor), 68 * 68 * 3, dim=1)
        std = torch.exp(log_var)
        image = torch.normal(mean=mu, std=std).to(device)
        image = image.clip(0,1).detach().cpu().numpy()
        image = image.reshape((68,68,3))
        ax[i].imshow(image)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    [ax[-j].set_visible(False) for j in range(1, len(ax)-i)]
    fig.suptitle("Tweaking isolated Latent variable, feature "+ str(j+1))
    plt.show()



