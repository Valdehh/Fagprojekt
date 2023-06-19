############################################################################################################
# local_dist.py
# This script is to plot some of the distributions of the latent space.
############################################################################################################




import torch
import matplotlib.pyplot as plt
import numpy as np
from dataloader import BBBC
from torch.utils.data import DataLoader
from semi_supervised_bbbc import Semi_supervised_VAE, encoder, decoder, classifier
from final_vae import VAE, decoder, encoder
from scipy.stats import norm

################################################################################################    
## DATA SECTION
################################################################################################

train_size = 100_000
test_size = 30_000


torch.backends.cudnn.deterministic = True       
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

from dataloader import BBBC

main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"    
main_path = "/Users/nikolaj/Fagprojekt/Data/"

exclude_dmso = False
shuffle = True
batch_size = 10

subset = (train_size, test_size)

dataset_train = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                    meta_path=main_path + "metadata.csv",
                    subset=subset,
                    test=False,
                    exclude_dmso=exclude_dmso,
                    shuffle=shuffle)

dataset_test = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                    meta_path=main_path + "metadata.csv",
                    subset=subset,
                    test=True,
                    exclude_dmso=exclude_dmso,
                    shuffle=shuffle)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

classes_, indexes = np.unique(dataset_test.meta[dataset_test.col_names[-1]], return_index=True)
boolean = len(classes_) == 13 if not exclude_dmso else len(classes_) == 12


################################################################################################
## TRAINING SECTION
################################################################################################



from final_vae import VAE, decoder, encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


vae = VAE(300, 68, 3, 0.1)

encoder_ = torch.load('/Users/nikolaj/Downloads/final_vae2/encoder.pt', map_location=device)
decoder_ = torch.load('/Users/nikolaj/Downloads/final_vae2/decoder.pt', map_location=device)

vae.encoder = encoder_
vae.decoder = decoder_


x =  np.linspace(-4.5, 4.5, 100)

for index in indexes:
    image = dataset_test[index]
    _, _, _, z = vae(image['image'].view(1,3,68,68), save_latent=True)
    # display mean and variance of z in bottom right corner
    plt.hist(z.detach().flatten().numpy(), bins=100, alpha=0.8, label='z', density=True)

    plt.xlim(-4.5, 4.5)
    
    plt.plot(x, norm.pdf(x, 0, 1), label='N(0,1)', linewidth=3, color='red')
    
    plt.title('Vanillia: ' + str(image['moa_name']), fontsize=20, fontweight='bold')
    
    plt.legend()

    plt.savefig('plots/dist/Vanilla_' + str(image['id']) + '.png', bbox_inches='tight')
    plt.show()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_ = torch.load('/Users/nikolaj/Downloads/semi_bbbc/encoder.pt', map_location=device)
decoder_ = torch.load('/Users/nikolaj/Downloads/semi_bbbc/decoder.pt', map_location=device)
classifier_ = torch.load('/Users/nikolaj/Downloads/semi_bbbc/classifier.pt', map_location=device)

middle_1 = torch.load('june_16/save_middle/middel_1.pt', map_location=device)
middle_2 = torch.load('june_16/save_middle/middel_2.pt', map_location=device)
middle_3 = torch.load('june_16/save_middle/middel_3.pt', map_location=device)


semi = Semi_supervised_VAE(13, 300, 68, 3, 0.1)
semi.classifier = classifier_
semi.decoder = decoder_
semi.encoder = encoder_

semi.middel_1 = middle_1
semi.middel_2 = middle_2
semi.middel_3 = middle_3





x =  np.linspace(-4.5, 4.5, 100)

for index in indexes:
    y = dataset_test[index]['moa']
    image = dataset_test[index]
    _, _, _, z = semi(image['image'].view(1,3,68,68), torch.tensor(y), save_latent=True)
    plt.hist(z.detach().flatten().numpy(), bins=100, alpha=0.8, label='z', density=True)
    plt.xlim(-4.5, 4.5)
    plt.plot(x, norm.pdf(x, 0, 1), label='N(0,1)', linewidth=3, color='red')
    plt.title('Semi: ' + str(image['moa_name']), fontsize=20, fontweight='bold')
    plt.legend()
    plt.savefig('plots/dist/Semi_' + str(image['id']) + '.png', bbox_inches='tight')
    plt.show()




