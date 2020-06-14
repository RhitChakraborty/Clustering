import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

#no. of clusters using Dendogram
import scipy.cluster.hierarchy as sch
# dendogram=sch.dendrogram(sch.linkage(x,method='ward'))  #single,complete,etc
# plt.title('Dendogram')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean Distance')
# plt.show()

#fitting Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)
print(y_hc)

#Visualising all points and cluster
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=20,c='red',label='Potential')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=20,c='blue',label='High value')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=20,c='green',label='High Spender')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=20,c='cyan',label='Low Value')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=20,c='magenta',label='Low Spender')
plt.title('Clusters of customers')
plt.xlabel('Annual income(k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

