import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Read the data
df = pd.read_csv("iris_data.txt", header = None)


# Shuffle data to split into train and test
np.random.seed(1)
new_df = df.iloc[np.random.permutation(len(df))]

x_train = new_df.iloc[:105,[0,1,2,3]]
y_train = new_df.iloc[:105,4]
x_test = new_df.iloc[105:,[0,1,2,3]]
y_test = new_df.iloc[105:,4]


# Build unsupervised model (k-means clustering)
pca = PCA(n_components=3).fit(new_df.iloc[:,[0,1,2,3]])
pca_2d = pca.transform(new_df.iloc[:,[0,1,2,3]])


# K-Means with 3 clusters
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(new_df.iloc[:,[0,1,2,3]])


# Scatter Plot
plt.figure('K-means with 3 clusters')
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
plt.show()