import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values
# print(x)

#no. of clusters using Elbow method
from sklearn.cluster import KMeans
# wcss=[]
# for i in range(1,11):
#     kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
#     kmeans.fit(x)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1,11),wcss)
# plt.title('WCSS')
# plt.xlabel('No. of Clusters')
# plt.ylabel('WCSS')
# plt.show()

#k means with optimum clusters
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)
#print(kmeans.cluster_centers_)
#print(dataset.iloc[:,2].values,y_kmeans)

#Visualising all points and cluster
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=20,c='red',label='Potential')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=20,c='blue',label='High value')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=20,c='green',label='High Spender')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=20,c='cyan',label='Low Value')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=20,c='magenta',label='Low Spender')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='yellow',label='Centroid')
plt.title('Clusters of customers')
plt.xlabel('Annual income(k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

