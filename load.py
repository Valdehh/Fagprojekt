############################################################################################################
# load.py
# This script is used to build a matrix of images and labels.
############################################################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Data\metadata.csv")

data = np.asarray(df)
attributes = np.array(df.columns)

x1 = np.load(os.path.join(
    "Data", "singh_cp_pipeline_singlecell_images", data[0, 2], data[0, 4]))

X = []
y = []
compound = []
for i in range(len(data)):
    file_path = os.path.join(
        "Data", "singh_cp_pipeline_singlecell_images", data[i, 2], data[i, 4])
    if os.path.exists(file_path):
        image = np.load(file_path)
        new_image = []
        for channel in range(image.shape[2]):
            max_val = np.max(image[:, :, channel])
            new_image.append(image[:, :, channel] / max_val)
        new_image = np.array(new_image, dtype=np.float16).transpose(1, 2, 0)
        X.append(new_image)
        y.append(data[i, -1])
        compound.append(data[i, -3])

X = np.array(X)
y = np.array(y)

np.savez("image_matrix.npz", images=X, labels=y, compound=compound)

print(X.shape)
print(y.shape)
