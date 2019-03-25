import pandas as pd
import matplotlib.pyplot as plt


# Data import
filename = input('Enter the name of a file: ')
dataset = pd.read_csv(filename, sep='[;\t,]', engine='python')
# print(dataset.head())

# Features' filtering
x = dataset.iloc[:, 1:].values

# Dataset visualization
plt.scatter(x[:, 0], x[:, 1], s=35)
plt.title('The original dataset')
plt.grid()
plt.show()
#plt.savefig('images/kmean01.png', dpi=300)

# Finding the optimum number of clusters for k-means classification - the elbow method
# Ideą tej metody jest obliczenie k-średnich klastrów w zbiorze danych dla zakresu wartości k (np. K od 1 do 20 w przykładzie ponizej), a dla każdej wartości k oblicza sumę kwadratów błędów (SSE):
from sklearn.cluster import KMeans
# import seaborn as sns; sns.set()  # for plot styling

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

print(wcss)
# Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (distortion)')  # within cluster sum of squares = SSE
plt.grid()
plt.show()
#plt.savefig('images/kmean02.png', dpi=300)


cluster_num = int(input('Enter the number of clusters: '))
# Applying k-means to the dataset / Creating the k-means classifier
kmeans = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)

# Visualising the clusters
#### Jak dodać etykiety? ###
# Cluster1 = Iris-setosa
# Cluster2 = Iris-versicolour
# Cluster3 = Iris-virginica
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=35, c='orange', edgecolor='black', label='Cluster1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=35, c='lightblue', edgecolor='black', label='Cluster2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=35, c='lightgreen', edgecolor='black', label='Cluster3')
#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red', edgecolor='black', label='Centroids')
title = input('Enter the title: ')
plt.title(title)
plt.legend()
plt.grid()
plt.show()
#plt.savefig('images/kmean03.png', dpi=300)




"""K-means starts with allocating cluster centers randomly and then looks for "better" solutions. K-means++ starts with allocation one cluster center randomly and then searches for other centers given the first one. So both algorithms use random initialization as a starting point, so can give different results on different runs. https://stats.stackexchange.com/questions/130888/k-means-vs-k-means"""