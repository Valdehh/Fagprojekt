import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


#load flat data
X = np.load("image_matrixflat.npz")["images"]
y = np.load("image_matrixflat.npz")["labels"]
compound = np.load("image_matrixflat.npz")["compound"]

#define pixel names
pixel = []
for i in range(len(X[1])):
    pixel.append('pixel'+str(i+1))

#create dataframe for anova test
df = pd.DataFrame(X)
df.columns = pixel
df.insert(len(X[1]), 'compound', compound)
df.insert(len(X[1])+1, 'labels', y)
df['labels'] = pd.Categorical(df['labels']).codes
df['compound'] = pd.Categorical(df['compound']).codes

#coubts of pixel where label or compound is insignificant
count_compound = 0
count_labels = 0

#doing anova for all pixels and keeping track of counts
for p in pixel:
    model = ols(f'{p} ~ compound + labels', data=df).fit()
    result = sm.stats.anova_lm(model, type=2)
    if result['PR(>F)'][0]>0.05:
        count_compound += 1
    if result['PR(>F)'][1]>0.05:
        count_labels += 1
    
print("number of pixels where labels are insignificant" + count_labels)
print("number of pixels where compound are insignificant" + count_compound)
