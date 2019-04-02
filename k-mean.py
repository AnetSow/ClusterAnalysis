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
# Obliczenie k-średnich klastrów w zbiorze danych dla zakresu wartości k, a dla każdej wartości k oblicza sumę kwadratów błędów (SSE):
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
print(wcss)

plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (distortion)')  # within cluster sum of squares = SSE
plt.grid()
plt.show()
#plt.savefig('images/kmean02.png', dpi=300)

cluster_num = int(input('Enter the number of clusters: '))


# Applying k-means to the dataset = k-means classifier
kmeans = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)


# Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=35, c='orange', edgecolor='black', label='Cluster1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=35, c='lightblue', edgecolor='black', label='Cluster2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=35, c='lightgreen', edgecolor='black', label='Cluster3')
# Cluster1 = Iris-setosa
# Cluster2 = Iris-versicolour
# Cluster3 = Iris-virginica


#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red', edgecolor='black', label='Centroids')
plt.title('K-means method')
plt.legend()
plt.grid()
plt.show()
#plt.savefig('images/kmean03.png', dpi=300)




"""K-means starts with allocating cluster centers randomly and then looks for "better" solutions. K-means++ starts with allocation one cluster center randomly and then searches for other centers given the first one. So both algorithms use random initialization as a starting point, so can give different results on different runs. https://stats.stackexchange.com/questions/130888/k-means-vs-k-means"""