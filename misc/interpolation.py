############################################################################################################
# interpolation.py
# This script is used to interpolate between images in the latent space (and create plots for the report).
############################################################################################################

import numpy as np
import torch
import sys
# Quick fix to solve the enviroment. Real world solution is to build a module and import it.
sys.path.append('.')
from DataLoader import get_image_based_on_id
import matplotlib.pyplot as plt

############################################################################################################
# HELPER FUNCTIONS
############################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def interpolate(v1, v2, Nstep):
        for i in range(Nstep):
            r = v2 - v1
            v = v1 + r * (i / (Nstep - 1))
            yield v

def interpolate_between_two_images(vae, index_image_1, index_image_2, main_path, results_folder=''):
    channels = vae.channels
    input_dim = vae.input_dim

    get_image = get_image_based_on_id(main_path + "singh_cp_pipeline_singlecell_images", main_path + "metadata.csv")
    _, _, _, z1 = vae(get_image[index_image_1]["image"].clone().detach().float().view(1,3,68,68).to(device), save_latent=True)
    _, _, _, z2 = vae(get_image[index_image_2]["image"].clone().detach().float().view(1,3,68,68).to(device), save_latent=True)


    generated_images = []

    for z in interpolate(z1, z2, 9):
        mean, var = vae.decode(z)
        image = torch.normal(mean=mean, std=torch.sqrt(var)).to(device)
        image = image.view(channels, input_dim, input_dim)
        image = image.clip(0,1).detach().cpu().numpy()  
        generated_images.append(image)

    fig, ax = plt.subplots(1, 11, figsize=(10, 2))
    ax[0].imshow(get_image[index_image_1]["image"].reshape((68,68,3)))
    ax[0].axis('off')
    ax[0].set_title("Original\nimage 1")
    for i in range(9):
        ax[i+1].imshow(generated_images[i].reshape((68,68,3)))
        ax[i+1].axis('off')
    ax[10].imshow(get_image[index_image_2]["image"].reshape((68,68,3)))
    ax[10].axis('off')
    ax[10].set_title("Original\nimage 2")
    title = "Interpolation between two images"
    title = ""
    print(get_image[index_image_1]["moa"], get_image[index_image_2]["moa"])
    if get_image[index_image_1]["moa"] == get_image[index_image_2]["moa"]:
        title += "(" + get_image[index_image_1]["moa_name"] + ")"
    # bold
    #fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    #plt.title("Interpolation between two images")
    plt.savefig(results_folder + str(index_image_1) + '_' + str(index_image_2) + '_unsup.png')
    plt.show()



def interpolate_between_three_images(vae, index_image_1, index_image_2, index_image_3, main_path, results_folder=''):
    channels = vae.channels
    input_dim = vae.input_dim

    get_image = get_image_based_on_id(main_path + "singh_cp_pipeline_singlecell_images", main_path + "metadata.csv")

    
    image_1 = get_image[index_image_1]["image"].clone().detach().float().view(1,3,68,68).to(device)
    image_2 = get_image[index_image_2]["image"].clone().detach().float().view(1,3,68,68).to(device)
    image_3 = get_image[index_image_3]["image"].clone().detach().float().view(1,3,68,68).to(device)

    ## plot the three images
    fig, ax = plt.subplots(1, 3, figsize=(10, 2))
    ax[0].imshow(image_1.reshape((68,68,3)))
    ax[0].axis('off')
    ax[0].set_title("Original\nimage 1")
    ax[1].imshow(image_2.reshape((68,68,3)))
    ax[1].axis('off')
    ax[1].set_title("Original\nimage 2")
    ax[2].imshow(image_3.reshape((68,68,3)))
    ax[2].axis('off')
    ax[2].set_title("Original\nimage 3")
    title = "Interpolation between three images"
    title = ""
    print(get_image[index_image_1]["moa"], get_image[index_image_2]["moa"], get_image[index_image_3]["moa"])
    if get_image[index_image_1]["moa"] == get_image[index_image_2]["moa"] == get_image[index_image_3]["moa"]:
        title += "(" + get_image[index_image_1]["moa_name"] + ")"
    # bold
    #fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    #plt.title("Interpolation between two images")
    plt.savefig(results_folder + str(index_image_1) + '_' + str(index_image_2) + '_' + str(index_image_3) + '.png')
    plt.show()



    _, _, _, z1 = vae(image_1, save_latent=True)
    _, _, _, z2 = vae(image_2, save_latent=True)
    _, _, _, z3 = vae(image_3, save_latent=True)


    def interpolate2(v1, v2, Nstep):
        for i in range(Nstep):
            r = v2 - v1
            v = r * (i / (Nstep - 1))
            yield v

    generated_images = [[] for _ in range(9)]
    
    

    for z_ in interpolate2(z1, z3, 9):
        for i, z in enumerate(interpolate(z1 + z_, z2 + z_, 6)):
            mean, var = vae.decode(z)
            image = torch.normal(mean=mean, std=torch.sqrt(var)).to(device)
            image = image.view(channels, input_dim, input_dim)
            image = image.clip(0,1).detach().cpu().numpy()  
            generated_images[i].append(image)
    
    fig, ax = plt.subplots(6, 9, figsize=(10, 6))
    for i in range(6):  
        for j in range(9):
            ax[i,j].imshow(generated_images[i][j].reshape((68,68,3)))
            ax[i,j].axis('off')
    fig.suptitle("Interpolation between three images", fontsize=16)
    plt.tight_layout()
    plt.savefig(results_folder + 'interpol_grid.png')
    plt.show()



def interpolate_between_two_images_semi(vae, index_image_1, index_image_2, main_path, results_folder=''):

    channels = vae.channels
    input_dim = vae.input_dim


    get_image = get_image_based_on_id(main_path + "singh_cp_pipeline_singlecell_images", main_path + "metadata.csv")

    y1 = torch.tensor(get_image[index_image_1]["moa"])
    y2 = torch.tensor(get_image[index_image_2]["moa"])

    assert y1 == y2
    y_hat = torch.zeros(13)
    y_hat[y1] = 1 # the one hot encoding of the moa

    _, _, _, z1 = vae(get_image[index_image_1]["image"].clone().detach().float().view(1,3,68,68).to(device), y1, save_latent=True)
    _, _, _, z2 = vae(get_image[index_image_2]["image"].clone().detach().float().view(1,3,68,68).to(device), y2, save_latent=True)


    generated_images = []

    for z in interpolate(z1, z2, 9):
        mean, var = vae.decode(z, y_hat.view(1,13)) #torch.ones(13).view(1,13)/13)  # dont know aboyt his
        image = torch.normal(mean=mean, std=torch.sqrt(var)).to(device)
        image = image.view(channels, input_dim, input_dim)
        image = image.clip(0,1).detach().cpu().numpy()  
        generated_images.append(image)

    fig, ax = plt.subplots(1, 11, figsize=(10, 2))
    ax[0].imshow(get_image[index_image_1]["image"].reshape((68,68,3)))
    ax[0].axis('off')
    ax[0].set_title("Original\nimage 1")
    for i in range(9):
        ax[i+1].imshow(generated_images[i].reshape((68,68,3)))
        ax[i+1].axis('off')
    ax[10].imshow(get_image[index_image_2]["image"].reshape((68,68,3)))
    ax[10].axis('off')
    ax[10].set_title("Original\nimage 2")
    title = "Interpolation between two images"
    title = ""
    print(get_image[index_image_1]["moa"], get_image[index_image_2]["moa"])
    if get_image[index_image_1]["moa"] == get_image[index_image_2]["moa"]:
        title += "(" + get_image[index_image_1]["moa_name"] + ")"
    plt.tight_layout()
    plt.savefig(results_folder + str(index_image_1) + '_' + str(index_image_2) + '_semi.png')
    plt.show()

############################################################################################################
# UN-SUPERVISED VAE - INTERPOLATION
############################################################################################################


from final_vae import VAE, encoder, decoder

vae = VAE(300, 68, 3, 0.1)

# load the model from final_vae with seed 42
encoder_ = torch.load('/Users/nikolaj/Downloads/final_vae2/encoder.pt', map_location=device)
decoder_ = torch.load('/Users/nikolaj/Downloads/final_vae2/decoder.pt', map_location=device)

vae.encoder = encoder_
vae.decoder = decoder_

main_path = "Data/" # path to singlecells folder and metadata

# these will create the plots included in the report. (given you have the decoder/encoder and the data)

interpolate_between_two_images(vae, 219669, 358837, main_path,'misc/plots/')

interpolate_between_two_images(vae, 308072, 328098, main_path,'misc/plots/')

interpolate_between_three_images(vae, 308072, 328098, 386683, main_path, 'misc/plots/')


############################################################################################################
# SEMI-SUPERVISED VAE - INTERPOLATION
############################################################################################################


from final_semi import Semi_supervised_VAE, encoder, decoder, classifier

encoder_ = torch.load('/Users/nikolaj/Downloads/semi_bbbc/encoder.pt', map_location=device)
decoder_ = torch.load('/Users/nikolaj/Downloads/semi_bbbc/decoder.pt', map_location=device)
classifier_ = torch.load('/Users/nikolaj/Downloads/semi_bbbc/classifier.pt', map_location=device)

middle_1 = torch.load('/Users/nikolaj/Desktop/GIT/test/BSUBS/june_16/save_middle/middel_1.pt', map_location=device)
middle_2 = torch.load('/Users/nikolaj/Desktop/GIT/test/BSUBS/june_16/save_middle/middel_2.pt', map_location=device)
middle_3 = torch.load('/Users/nikolaj/Desktop/GIT/test/BSUBS/june_16/save_middle/middel_3.pt', map_location=device)

main_path = "Data/"

vae = Semi_supervised_VAE(13, 300, 68, 3, 0.1)
vae.classifier = classifier_
vae.decoder = decoder_
vae.encoder = encoder_

vae.middel_1 = middle_1
vae.middel_2 = middle_2
vae.middel_3 = middle_3

interpolate_between_two_images_semi(vae, 219669, 358837, main_path,'misc/plots/')

interpolate_between_two_images_semi(vae, 308072, 328098, main_path,'misc/plots/')



