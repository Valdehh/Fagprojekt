import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm
import os
from DataLoader import BBBC
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from vae_semi_supervised import classifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classifier_semi = torch.load('/Users/nikolaj/Downloads/semi_bbbc/classifier.pt', map_location=device)
classifier_basic = torch.load('final_SC.pt', map_location=device)

train_size = 100_000
test_size = 30_000
batch_size = 100

# latent_dim = 10
#epochs, batch_size, train_size, test_size = 2, 10, 10, 10

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"
main_path = "Data/"



exclude_dmso = True
shuffle = True

subset = (train_size, test_size)


dataset_test = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                        meta_path=main_path + "metadata.csv",
                        subset=subset,
                        test=True,
                        exclude_dmso=exclude_dmso,
                        shuffle=shuffle)



loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)


list_of_compounds = np.unique(dataset_test.meta[dataset_test.col_names[-3]])
assert len(list_of_compounds) == 38

real_label = np.zeros((len(dataset_test)))


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    save_pred = np.zeros((len(test_loader.dataset)))
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            data = batch['image'].to(device)
            target = batch['moa'].to(device).long()
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            save_pred[i*batch_size: (i+1)*batch_size] = pred.detach().cpu().numpy()
            correct += (pred == target).sum().item()
            real_label[i*batch_size: (i+1)*batch_size] = target.detach().cpu().numpy()
    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset), save_pred


test_loss_semi, accuracy_semi, save_pred_semi = test(model=classifier_semi, 
                                                     test_loader=loader_test, 
                                                     criterion=torch.nn.CrossEntropyLoss(), 
                                                     device=device)

test_loss_basic, accuracy_basic, save_pred_basic = test(model=classifier_basic,
                                                        test_loader=loader_test,
                                                        criterion=torch.nn.CrossEntropyLoss(),
                                                        device=device)



print(f"Test loss semi: {test_loss_semi:.4f}, accuracy semi: {accuracy_semi:.4f}")
print(f"Test loss basic: {test_loss_basic:.4f}, accuracy basic: {accuracy_basic:.4f}")


pred_semi = save_pred_semi
pred_basic = save_pred_basic

#get contigency table and perform mcnemar
contingency_table = np.zeros((2,2))
for i in range(len(real_label)):
    if pred_basic[i] == real_label[i] and pred_semi[i] == real_label[i]:
        contingency_table[0,0] += 1
    if pred_basic[i] != real_label[i] and pred_semi[i] != real_label[i]:
        contingency_table[1,1] += 1
    if pred_basic[i] == real_label[i] and pred_semi[i] != real_label[i]:
        contingency_table[0,1] += 1
    if pred_basic[i] != real_label[i] and pred_semi[i] == real_label[i]:
        contingency_table[1,0] += 1

print(contingency_table)

test_results = mcnemar(contingency_table, exact=True)

print(test_results)
