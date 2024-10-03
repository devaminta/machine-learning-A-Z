# hierarchical clustering


# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,[3,4]].values


# using the dendrogram to find the optimal number of clusters

import scipy.cluster.hierarchy as sch

# dendogram=sch.dendrogram(sch.linkage(X,method="ward"))
# # plt.title("dendrogram")
# # plt.xlabel("Customers")
# # plt.ylabel("Euclidean Distance")
# # plt.show()


# Fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=5,linkage="ward")
y_hc=hc.fit_predict(X)
#  Visualizing the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c="red",label="Cluster1")
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c="blue",label="Cluster2")
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c="green",label="Cluster3")
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c="cyan",label="Cluster4")
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c="magenta",label="Cluster5")
plt.title("Clusters of clients")
plt.xlabel("Annual income($)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

