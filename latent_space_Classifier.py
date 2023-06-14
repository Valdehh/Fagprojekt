import torch
from sklearn.svm import SVC
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import scipy.stats as st
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

from statsmodels.stats.contingency_tables import mcnemar

#load data
X_basic = np.load("latent_space_vanilla.npz")["z"]
y = np.load("latent_space_vanilla.npz")["labels"]
compound = np.load("latent_space_vanilla.npz")["compound"]
'''
X_semi = np.load("latent_space_semi.npz")["z"]
'''

index = np.where(y != 'DMSO')


y = y[index]

compound = compound[index]
X_basic=X_basic[index]

def label_encoder(x):
    classes = np.array([ 'Actin disruptors', 'Aurora kinase inhibitors',
    'Cholesterol-lowering', 'DNA damage', 'DNA replication',
    'Eg5 inhibitors', 'Epithelial', 'Kinase inhibitors',
    'Microtubule destabilizers', 'Microtubule stabilizers',
    'Protein degradation', 'Protein synthesis'])
    
    return np.where(classes == x)[0][0]

for i in range(len(y)):
    y[i]=label_encoder(y[i])

#get unique compunds
list_of_compounds = np.unique(compound)

#define SVC
model_basic = SVC(kernel='linear', gamma='scale')
model_semi = SVC(kernel='linear', gamma='scale')

#Define lists for 
score_basic = []
score_semi = []

pred_basic = np.array(['0'])
pred_semi = []

num_data=[]

real_label = np.array(['0'])

# Define feature sensitivities
feature_sensitivity_basic = []
feature_sensitivity_semi = []

for Unique_Compound in list_of_compounds:
    #find compound
    test_comp = np.where(compound == Unique_Compound)
    train_comp = np.where(compound != Unique_Compound)

    #index labels
    y_test = y[test_comp]
    y_train = y[train_comp]

    num_data.append(len(y_test)/len(y))

    #index basic VAE train and test
    X_basic_test = X_basic[test_comp]
    X_basic_train = X_basic[train_comp]


    #fit models
    model_basic.fit(X_basic_train, y_train)
    predict=model_basic.predict(X_basic_test)
    pred_basic = np.hstack((pred_basic, predict))

    print(pred_basic.shape)

    #append scores and test labels
    score_basic.append(model_basic.score(X_basic_test, y_test)) 
    real_label = np.hstack((real_label,y_test))  
    print(real_label.shape)
    '''
    #compute decision function and compute gradients and store
    X_basic_test_tensor=torch.tensor(X_basic_test, device=device, dtype=torch.float32, requires_grad=True)
    output_basic = torch.tensor(model_basic.decision_function(X_basic_test), device=device, dtype=torch.float32, requires_grad=True)
    output_basic.backward()
    gradients_basic = X_basic_test_tensor.grad.numpy()
    
    feature_sensitivity_basic.append(gradients_basic.mean(axis=0))
    '''
    '''
    #index semi VAE train and test
    X_semi_train = X_semi[train_comp]
    X_semi_test = X_semi[test_comp]

    #fit models
    model_semi.fit(X_semi_train, y_train)
    pred_semi.append(model_semi.predict(X_semi_test))

    #append scores and test labels
    score_semi.append(model_semi.score(pred_semi, y_test))

    # calculate gradients for semi model
    X_semi_test_tensor=torch.tensor(X_semi_test, device=device, dtype=torch.float32, requires_grad=True)
    output_semi = torch.tensor(model_semi.decision_function(X_semi_test), device=device, dtype=torch.float32, requires_grad=True)
    output_semi.backward(torch.ones_like(output_semi))
    gradients_semi = X_semi_test_tensor.grad
    feature_sensitivity_semi.append(gradients_semi.mean(axis=0).numpy())
    '''
pred_basic=pred_basic[1:]
real_label=real_label[1:]

num_data = np.array(num_data)
score_basic = np.array(score_basic)

score=np.sum(num_data*score_basic)
#define and plot cf-matix and plot
'''
Confusion_matrix_basic = confusion_matrix(pred_basic, real_label, normalize='true')
#Confusion_matrix_semi = confusion_matrix(pred_semi, real_label)

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
'''