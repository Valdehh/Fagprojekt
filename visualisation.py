import numpy as np
import matplotlib.pyplot as plt

data = np.load('image_matrix.npz')
images = data['images']
labels = data['labels']

print(images.shape)
print(labels.shape)

MOAs, counts = np.unique(labels, return_counts=True)

[print(MOAs[i], ":", counts[i]) for i in range(len(MOAs))]

plt.imshow(images[0])
plt.xticks([])
plt.xticks([])
plt.savefig('single-cell.png')
plt.show()

fig, ax = plt.subplots(5, 3)
ax = ax.flatten()
for i, MOA in enumerate(MOAs):
    image = images[labels == MOA][0]
    ax[i].imshow(image)
    ax[i].set_title(MOA)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
[ax[-j].set_visible(False) for j in range(1, len(ax)-i)]
fig.suptitle("Cell image for each kind of MOA")
plt.show()
