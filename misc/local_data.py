############################################################################################################
# local_data.py
# This script is to plot some of the training data (used in data section). 
############################################################################################################


import torch
import numpy as np
import matplotlib.pyplot as plt

train_size = 100_000
test_size = 10_000

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

from DataLoader import BBBC

main_path = "Data/"

exclude_dmso = False
shuffle = True

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

classes_, indexes = np.unique(dataset_train.meta[dataset_train.col_names[-1]], return_index=True)
boolean = len(classes_) == 13 if not exclude_dmso else len(classes_) == 12


# fig, ax = plt.subplots(3,5, figsize=(15,10))

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 14)
fig = plt.figure(figsize=(21, 6))

for i in range(7):
    ax1 = plt.subplot(gs[0, i*2:(i+1)*2], )
    ax1.imshow(dataset_train[indexes[i]]['image'].reshape(68,68,3))
    ax1.axis('off')
    ax1.set_title(dataset_train[indexes[i]]['moa_name'], fontsize=14, fontweight='bold')    

for i in range(6):
    ax2 = plt.subplot(gs[1, (i*2+1):((i+1)*2+1)])
    ax2.imshow(dataset_train[indexes[i+7]]['image'].reshape(68,68,3))
    ax2.axis('off')
    ax2.set_title(dataset_train[indexes[i+7]]['moa_name'], fontsize=14, fontweight='bold')
gs.tight_layout(fig)
# plt.savefig('images_from_train_set.png', dpi=300)
print("The plot shown isn't the same as the plot saved. The saved plot is correct.!!")
plt.show()