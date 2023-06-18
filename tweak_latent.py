import numpy as np
import torch
import numpy as np
import torch
import sys
sys.path.insert(1, 'C:/Users/andre/OneDrive/Skrivebord/Fagprojekt')
from final_vae import decoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
model_mid = torch.load('C:/Users/andre/OneDrive/Skrivebord/Fagprojekt/middel_3.pt', map_location=device)
model = torch.load('C:/Users/andre/OneDrive/Skrivebord/Fagprojekt/decoder_semi.pt', map_location=device)

latent = np.load('latent_space_semi.npz')['z']

#18

features = [8, 32, 35, 38, 45, 74, 78, 148, 129, 148, 184, 192, 196, 253, 289, 123, 298]

for j in range(300):
    val=np.linspace(np.min(latent[:,j])*2, 2*np.max(latent[:,j]), num=16)
    latent_tensor = torch.tensor(latent[0], device=device, dtype=torch.float)
    latent_tensor[0]=val[0]

    print(np.max(latent[:,j]))
    print(np.min(latent[:,j]))
    print(latent_tensor[j])
    
    fig, ax = plt.subplots(4, 4)
    ax = ax.flatten()
    for i in range(16):
        latent_tensor[j] = val[i]
        print(latent_tensor[j])
        z = model_mid(torch.cat((latent_tensor.view(1,300), torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0]])), dim=1))
        mu, log_var = torch.split(
            model.forward(z), 68 * 68 * 3, dim=1)
        std = torch.exp(log_var)
        image = torch.normal(mean=mu, std=std).to(device)
        image = image.clip(0,1).detach().cpu().numpy()
        image = image.reshape((68,68,3))
        ax[i].imshow(image)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    [ax[-j].set_visible(False) for j in range(1, len(ax)-i)]
    fig.suptitle("Tweaking isolated Latent variable, feature "+ str(j+1))
    plt.tight_layout()
    plt.show()


def label_encoder(x):
    classes = np.array(['DMSO', 'Actin disruptors', 'Aurora kinase inhibitors',
    'Cholesterol-lowering', 'DNA damage', 'DNA replication',
    'Eg5 inhibitors', 'Epithelial', 'Kinase inhibitors',
    'Microtubule destabilizers', 'Microtubule stabilizers',
    'Protein degradation', 'Protein synthesis'])
    
    return np.where(classes == x)[0][0]
    
def label_decoder(x):
    classes = np.array(['DMSO', 'Actin disruptors', 'Aurora kinase inhibitors',
    'Cholesterol-lowering', 'DNA damage', 'DNA replication',
    'Eg5 inhibitors', 'Epithelial', 'Kinase inhibitors',
    'Microtubule destabilizers', 'Microtubule stabilizers',
    'Protein degradation', 'Protein synthesis'])
    
    return classes[x]







