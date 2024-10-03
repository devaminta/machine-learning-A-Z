# k-mean clustering


# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# importing the dataset

dataset=pd.read_csv("Mall_Customers.csv")

X=dataset.iloc[:,[3,4]].values

# using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    k_means=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
    k_means.fit(X)
    wcss.append(k_means.inertia_)
# plt.plot(range(1,11),wcss)
# plt.title("The Elbow Method")
# plt.xlabel("The number of Clusters")
# plt.ylabel("WCSS")
# plt.show()

# Applying k-means to the dataset

k_means=KMeans(n_clusters=5,init="k-means++",max_iter=300,n_init=10,random_state=0)
y_k_means=k_means.fit_predict(X)
print(y_k_means)

# Visualizing the clusters
plt.scatter(X[y_k_means==0,0],X[y_k_means==0,1],s=100,c="red",label="Cluster1")
plt.scatter(X[y_k_means==1,0],X[y_k_means==1,1],s=100,c="blue",label="Cluster2")
plt.scatter(X[y_k_means==2,0],X[y_k_means==2,1],s=100,c="green",label="Cluster3")
plt.scatter(X[y_k_means==3,0],X[y_k_means==3,1],s=100,c="cyan",label="Cluster4")
plt.scatter(X[y_k_means==4,0],X[y_k_means==4,1],s=100,c="magenta",label="Cluster5")
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=300,c="yellow",label="Centroids")
plt.title("Clusters of clients")
plt.xlabel("Annual income($)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()


   