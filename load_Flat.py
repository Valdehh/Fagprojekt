import os
import numpy as np
import pandas as pd



df = pd.read_csv("C:/Users/andre/Desktop/singlecell/metadata.csv")

data = np.asarray(df)
attributes = np.array(df.columns)

x1 = np.load(os.path.join(
    "C:/Users/andre/Desktop/singlecell", "singh_cp_pipeline_singlecell_images", data[0, 2], data[0, 4]))

X = []
y = []
compound = []
for i in range(len(data)):
    file_path = os.path.join(
        "C:/Users/andre/Desktop/singlecell", "singh_cp_pipeline_singlecell_images", data[i, 2], data[i, 4])
    if os.path.exists(file_path):
        image = np.load(file_path)
        min_val, max_val = np.min(image), np.max(image)
        pixel_range = max_val - min_val
        scaled_pixels = (image - min_val) / pixel_range
        scaled_pixels *= 255
        converted_image = np.round(scaled_pixels).astype(np.uint8)
        X.append(converted_image)
        y.append(data[i, -1])
        compound.append(data[i, -3])

X = np.array(X)
y = np.array(y)
compound = np.array(compound)

index = np.where(y != 'DMSO')

X=X.reshape(488396, 68*68*3)

y = y[index]
compound = compound[index]
X=X[index]

np.savez("image_matrixflat.npz", images=X, labels=y, compound=compound)