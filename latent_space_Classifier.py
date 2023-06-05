import torch
from sklearn.svm import SVC
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import scipy.stats as st
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar

#load data
X_basic = np.load("latent_space_basic.npz")["Z"]
y = np.load("latent_space_basic.npz")["labels"]
compound = np.load("latent_space_basic.npz")["compound"]
X_semi = np.load("latent_space_semi.npz")["Z"]

#get unique compunds
list_of_compounds = np.unique(compound)

#define SVC
model_basic = SVC(kernel='rbf', gamma='scale')
model_semi = SVC(kernel='rbf', gamma='scale')

#Define lists for 
score_basic = []
score_semi = []

pred_basic = []
pred_semi = []

real_label = []

for Unique_Compound in list_of_compounds:
    #find compound
    test_comp = np.where(compound == Unique_Compound)
    train_comp = np.where(compound != Unique_Compound)

    #index labels
    y_test = y[test_comp]
    y_train = y[train_comp]

    #index basic VAE train and test
    X_basic_test = X_basic[test_comp]
    X_basic_train = X_basic[train_comp]

    #index semi VAE train and test
    X_semi_train = X_semi[train_comp]
    X_semi_test = X_semi[test_comp]

    #fit models
    model_basic.fit(X_basic_train, y_train)
    pred_basic.append(model_basic.predict(X_basic_test))
    
    model_semi.fit(X_semi_train, y_train)
    pred_semi.append(model_semi.predict(X_semi_test))

    #append scores and test labels
    score_basic.append(model_basic.score(pred_basic, y_test)) 
    score_semi.append(model_semi.score(pred_semi, y_test))
    real_label.append(y_test)

#define and plot cf-matix and plot
Confusion_matrix_basic = confusion_matrix(pred_basic, y_test)
Confusion_matrix_semi = confusion_matrix(pred_semi, y_test)

plt.imshow(Confusion_matrix_basic, cmap='binary', interpolation='None')
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(np.arange(len(np.unique(y))), np.unique(y))
plt.yticks(np.arange(len(np.unique(y))), np.unique(y))
plt.title("Confusion matrix Basic")
plt.colorbar()
plt.show()

plt.imshow(Confusion_matrix_semi, cmap='binary', interpolation='None')
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(np.arange(len(np.unique(y))), np.unique(y))
plt.yticks(np.arange(len(np.unique(y))), np.unique(y))
plt.title("Confusion matrix semi-supervised")
plt.colorbar()
plt.show()


#get contigency table and perform mcnemar
contingency_table = np.zeros(2,2)
for i in range(len(real_label)):
    if pred_basic[i] == real_label[i] and pred_semi[i] == real_label[i]:
        contingency_table[0,0] += 1
    if pred_basic[i] != real_label[i] and pred_semi[i] != real_label[i]:
        contingency_table[1,1] += 1
    if pred_basic[i] == real_label[i] and pred_semi[i] != real_label[i]:
        contingency_table[0,1] += 1
    if pred_basic[i] != real_label[i] and pred_semi[i] == real_label[i]:
        contingency_table[1,0] += 1

test_results = mcnemar(contingency_table, exact=True)

print(test_results)
