import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import kruskal
import scipy.stats as stats
import pylab

results_folder = 'final_grid/'

latent_dims = [100, 200, 300, 400, 500]

save = []

for latent_dim in latent_dims:
    filename = str(latent_dim)+'/test_ELBOs.npz'
    data = np.load(results_folder+'/'+filename)

    save.append(data['test_ELBOs'])


# qq-plots by importing statsmodels.api as sm and using sm.qqplot(data, line='45')
# qq-plots for the different latent dimensions

import matplotlib.pyplot as plt
import statsmodels.api as sm

# Your five distributions (replace with your actual data)

# Create a figure with 5 subplots
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(24, 4))

# Generate QQ-plots for each distribution
for i, dist in enumerate(save):
    ax = axes[i]
    sm.qqplot(dist, line='s', ax=ax)
    ax.set_title(f'L= {latent_dims[i]}', fontsize=22, fontweight='bold')
    ax.set_yticklabels([])
    if i == 0:
        ax.set_ylabel('Quantiles')
    else:
        ax.set_ylabel(' ')


# Display the plot
plt.tight_layout()
plt.savefig(results_folder + 'qq_plots.png')
plt.show()

print('as they are somewhat normally distributed, we can use ANOVA to compare the means of the groups \n')


print(f_oneway(save[0], save[1], save[2], save[3]))
print('\n as the above p-value is very small, there must be a difference between the means of the groups')
print(' the following t-tests show which groups are different from each other \n')

# doing benferroni correction for multiple testing
print('t-tests with benferroni correction for multiple testing: \n')

p_vals = []

for i in range(len(latent_dims)):
    for j in range(i+1, len(latent_dims)):
        p_vals.append(ttest_ind(save[i], save[j])[1])

p_vals = np.array(p_vals)

k = 0

for i in range(len(latent_dims)):
    for j in range(i+1, len(latent_dims)):
        print('-'*20)
        print('t-test between L='+str(latent_dims[i])+' and L='+str(latent_dims[j])+':')
        print(p_vals[k] < 0.05/6) 
        print(p_vals[k])
        print("\n")
        k+=1



for i in range(len(latent_dims)):
    print('L='+str(latent_dims[i])+': '+str(np.mean(save[i])))

print('\n we see a significant difference between 300 and the others; therefore we choose this one as it is also the smallest one \n')
