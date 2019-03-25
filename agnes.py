"""Agglomerative clustering = AGNES (Agglomerative Nesting)"""

import pandas as pd
import matplotlib.pyplot as plt


# Data import
filename = input('Enter the name of a file: ')
dataset = pd.read_csv(filename, sep='[;\t,]', engine='python')

# print(dataset.head()) # categorical variables=labels

# Features' filtering
x = dataset.iloc[:, 1:].values

# Dataset visualization
plt.scatter(x[:, 0], x[:, 1], s=35)
plt.title('The original dataset')
plt.grid()
plt.show()
#plt.savefig('images/agnes01.png', dpi=300)



# Creating the dendrogram for the dataset
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt


linkage_matrix = linkage(x, 'single')
labelList = range(1, 11)
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.grid()
plt.show()
#plt.savefig('images/agnes02.png', dpi=300)


# Grouping the data points into clusters
from sklearn.cluster import AgglomerativeClustering


cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single')
cluster.fit_predict(x)

plt.figure(figsize=(10, 7))
plt.scatter(x[:, 0], x[:, 1], c=cluster.labels_, cmap='rainbow')
plt.grid()
plt.show()
