#%%
from ucimlrepo import fetch_ucirepo 
import importlib_resources
import numpy as np
import pandas as pd
  
# fetch dataset 
abalone = fetch_ucirepo(id=1) 
# data (as pandas dataframes) fafa
X = abalone.data.features 
y = abalone.data.targets 

raw_data = X.values
# metadata 
print(abalone.metadata) 
  
# variable information 
print(abalone.variables) 

cols = range(0, 8)
attributeNames = np.asarray(X.columns[cols]).tolist()

classLabels = y.values[:, 0]
classNames = np.unique(classLabels).tolist()
classDict = dict(zip(classNames, range(len(classNames))))

N, M = X.shape

C = len(classNames)



# Extract vector y, convert to NumPy matrix and transpose
yy = np.array([classDict[value] for value in classLabels])

# Preallocate memory, then extract data to matrix X
xx = np.empty((N, M))
for i in range(M):
    if i != 0:
        xx[:, i] = np.array(X.iloc[:,i]).T
    else:
        for j in range(N):
            if X.iloc[j, i] == 'M': #sex = 0 for males
                xx[j, i] = 0
            elif X.iloc[j, i] == 'F': #sex = 1 for females
                xx[j, i] = 1
            else:
                xx[j, i] = 2 #sex = 2 for infants

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 7))
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(xx[:, i], color=(0.2, 0.8 - i/4 * 0.2, 0.4))
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N / 2)

plt.show()
# %% Standardize the data
# Subtract the mean from the data
x_normalized = np.empty((N, M))
for i in range(M):
    x_normalized[:, i] = (xx[:, i] - np.mean(xx[:, i]))/np.std(xx[:, i])
# %%
plt.figure()
plt.boxplot(x_normalized)
plt.xticks(range(1, 9), attributeNames, rotation=60)
#plt.ylabel("cm")
plt.title("Abalone boxplot")
plt.show()
print("Ran Exercise 2.3.3")

# %%
