import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Data
X = np.array(pd.read_csv('data.csv'))
print(X.shape)

#Elbow
ls = []
n = 8

for i in range(1, n):
    k_means = KMeans(n_clusters=i, init='random')
    k_means.fit(X)
    ls.append(k_means.inertia_)
    
plt.plot(range(1,n), ls)
plt.title('Elbow')
plt.xlabel('Clusters')
plt.ylabel('Inertias')
plt.show()

#Model
k_means = KMeans(n_clusters=3, init='random')
X_pred = k_means.fit_predict(X)
centroids =  k_means.cluster_centers_

#Graph
plt.scatter(X[X_pred == 0, 0], X[X_pred == 0, 1], s=15, c='r', alpha='.6', label='Cluster 1')
plt.scatter(centroids[0,0],centroids[0,1],s=300, color='r', marker='X')
plt.scatter(X[X_pred == 1, 0], X[X_pred == 1, 1], s=15, c='g', alpha='.6', label='Cluster 2')
plt.scatter(centroids[1,0],centroids[1,1],s=300, color='g', marker='X')
plt.scatter(X[X_pred == 2, 0], X[X_pred == 2, 1], s=15, c='b', alpha='.6', label='Cluster 3')
plt.scatter(centroids[2,0],centroids[2,1],s=300, color='b', marker='X')
plt.legend()