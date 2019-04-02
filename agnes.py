"""Agglomerative clustering = AGNES (Agglomerative Nesting)"""

import pandas as pd
import matplotlib.pyplot as plt


# Data import
# filename = input('Enter the name of a file: ')
# dataset = pd.read_csv(filename, sep='[;\t,]', engine='python')
dataset = pd.read_csv('iris.csv', sep='[;\t,]', engine='python')


# Features' filtering
x = dataset.iloc[:, 1:].values
print('Dataset\n', x)


# Distance matrix
from scipy.spatial.distance import pdist, squareform

dist_matrix = pd.DataFrame(squareform(pdist(x, metric='euclidean')))
print('Distance matrix\n', dist_matrix)
"""
# Dataset visualization
plt.scatter(x[:, 0], x[:, 1], s=35)
plt.title('The original dataset')
plt.grid()
plt.show()
#plt.savefig('images/agnes01.png', dpi=300)
"""


# Creating the dendrogram for the dataset
# Applying the single linkage agglomeration to clusters
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt


row_clusters = linkage(pdist(dist_matrix, metric='euclidean'), method='single')
linkage_matrix = pd.DataFrame(row_clusters, columns=['row label 1', 'row label 2', 'distance', 'no of items in cluster'], index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])])
print('Linkage matrix\n', linkage_matrix)


row_dendr = dendrogram(row_clusters)
plt.tight_layout()
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Euclidean distance')
plt.show()

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
