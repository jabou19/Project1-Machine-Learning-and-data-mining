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
plt.figure(figsize=(16, 10))
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)
for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(xx[:, i], color=(0.2, 0.8 - i/4 * 0.2, 0.4))
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N / 1.3)

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
plt.title("Standardized Abalone boxplot")
plt.show()

print("Ran Exercise 2.3.3")


# %%

f, ax = plt.subplots(4,C//4, figsize=(14, 40))

for c in range(C):
    
    class_mask = yy == c # binary mask to extract elements of class c
    # or: class_mask = nonzero(y==c)[0].tolist()[0] # indices of class c

    ax.ravel()[c].boxplot(x_normalized[class_mask, :])
    # title('Class: {0}'.format(classNames[c]))
    ax.ravel()[c].set_title("Class: " + str(classNames[c]))
    # ax.ravel()[c].xticks(
    #     range(1, len(attributeNames) + 1), [a[:7] for a in attributeNames], rotation=45
    # )
    ax.ravel()[c].set_xticklabels(attributeNames, rotation=90)  # Rotate x-ticks
    #y_up = x_normalized.max() + (x_normalized.max() - x_normalized.min()) * 0.1
    #y_down = x_normalized.min() - (x_normalized.max() - x_normalized.min()) * 0.1
    #ax.ravel()[c].set_ylim(y_down, y_up)

plt.show()

# %%
plt.figure(figsize=(12, 10))
for m1 in range(M):
    for m2 in range(M):
        plt.subplot(M, M, m1 * M + m2 + 1)
        for c in range(C):
            class_mask = yy == c
            plt.plot(np.array(x_normalized[class_mask, m2]), np.array(x_normalized[class_mask, m1]), ".")
            if m1 == M - 1:
                plt.xlabel(attributeNames[m2])
            else:
                plt.xticks([])
            if m2 == 0:
                plt.ylabel(attributeNames[m1])
            else:
                plt.yticks([])
                            
plt.legend(classNames)

plt.show()
# %%
from scipy.linalg import svd 

# Subtract mean value from data
# Note: Here we use Y to in teh book we often use X with a hat-symbol on top.
Y = xx - np.ones((N, 1)) * xx.mean(axis=0)

# PCA by computing SVD of Y
# Note: Here we call the Sigma matrix in the SVD S for notational convinience
U, S, Vh = svd(Y, full_matrices=False)

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T

# Compute variance explained by principal components 
# Note: This is an important equation, see Eq. 3.18 on page 40 in the book.
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()
# %%
