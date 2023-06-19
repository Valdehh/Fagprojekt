# Fagprojekt



*Relevant files for this project*
## ANOVA_pixel.py
- This script is used to test what effect compoud and label has on each pixel.

## DataLoader.py
- This script contains the main dataloader used in the project. (especially when run on the cluster).

## latent_space_Classifier.py
- This script is used to build our classifier and perform the statistical tests.

## CNN_classifier.py
- This script contains the classifier used to directly classify the images.

## grid_search.py
- This script is used search for the optimal latent dimension for the VAE. 

## load_FLat.py
- This script is used to build a matrix of images and labels in a flat format.

## load.py
- This script is used to build a matrix of images and labels.

## tweak_latent.py
- This script is used to tweak the latent space of the VAE and plot the corresponding images.

## vae_semi_supervised.py
- This script contains the semi-supervised VAE.

## vae_unsupervised.py
- This script contains the un-supervised VAE.

## visualisation.py
- This script is to plot some of the training data (used in data section). 

## misc/interpolation.py
- This script is used to interpolate between images in the latent space (and create plots for the report).

## misc/local_convergence.py
- This script is to calculate the test ELBOs for the two models.
- 
## misc/local_data.py
- This script is to plot some of the training data (used in data section). 
  
## misc/local_dist.py
- This script is to plot some of the distributions of the latent space.
  
## misc/local_plot.py
- This script is to plot some of the training data (used in results section).

## statistical_tests\anova.R
- This script contains the anova in relation to the robustess analysis

## statistical_tests\t-test.R
- This script contains the t-tests for the grid-search

## statistical_tests\t-test.py
- This script also contains the t-tests for the grid-search and some more (plotting)


