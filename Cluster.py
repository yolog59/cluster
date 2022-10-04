from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, ward
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
import numpy as np
data = pd.read_table('4.txt', header = None, index_col = None, sep = '\s+')
z = linkage(data, method='centroid', metric = 'euclidean')
R = dendrogram(z)
plt.show()
plt.scatter(data[0], data[1])
plt.show()

clusters = fcluster(z, criterion = 'maxclust', t = 35)
plt.scatter(data[0], data[1], c = clusters, cmap = 'prism')
plt.show()

# 2 часть
kmeans = KMeans(n_clusters=35, random_state=1).fit(data)
y_kmeans = kmeans.predict(data)
plt.scatter(data[0], data[1], c=y_kmeans, s=35, cmap='prism')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=40, alpha=0.5)
plt.show()


cluster_labels = kmeans.labels_

sample_silhouette_values = silhouette_samples(data, cluster_labels)

means_lst = []
for label in range(20, 40):
    means_lst.append(sample_silhouette_values[cluster_labels == label].mean())
print(means_lst) 

list = np.arange(20, 40)
plt.scatter(list, means_lst)
plt.show()