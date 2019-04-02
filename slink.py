"""Metoda taksonomii wrocławskiej = met. najbliższego sąsiedztwa = met. najbliższej odległości = met. pojedynczego wiązania (ang. single linkage). """

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import pandas as pd


# Data import
# filename = input('Enter the name of a file: ')
# dataset = pd.read_csv(filename, sep='[;\t,]', engine='python')
dataset = pd.read_csv('iris.csv', sep='[;\t,]', engine='python')

# Features' filtering
x = dataset.iloc[:, 1:].values

# Dataset visualization
plt.scatter(x[:, 0], x[:, 1], s=35)
plt.title('The original dataset')
plt.grid()
plt.show()
#plt.savefig('images/slink01.png', dpi=300)


Z = linkage(x, 'single')

fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
plt.grid()
plt.title('Metoda taksonomii wrocławskiej')
plt.show()
#plt.savefig('images/slink02.png', dpi=300)

# # Validation
# M = x.as_matrix()
# # generate the linkage matrix
# single_link = linkage(M, 'single')  # using single link metric to evaluate 'distance' between clusters

