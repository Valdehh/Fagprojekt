import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Data\metadata.csv")

data = np.asarray(df)
attributes = np.array(df.columns)

# print(data[0])
# print(attributes)

# print(data[0, 2])
# print(data[0, 4])
# print(data[0, -1])

x1 = np.load(os.path.join(
    "Data", "singh_cp_pipeline_singlecell_images", data[0, 2], data[0, 4]))

X = []
y = []
for i in range(len(data)):
    file_path = os.path.join(
        "Data", "singh_cp_pipeline_singlecell_images", data[i, 2], data[i, 4])
    if os.path.exists(file_path):
        x = np.load(file_path)
        X.append(x)
        y.append(data[i, -1])

# X = [np.load(os.path.join("Data", "singh_cp_pipeline_singlecell_images", data[i, 2], data[i, 4])) for i in range(len(data))]

X = np.array(X)
y = np.array(y)

# Complete_data_matrix = np.hstack((X, y[:, None]))

np.savez("image_matrix.npz", images=X, labels=y)

print(X.shape)
print(y.shape)
