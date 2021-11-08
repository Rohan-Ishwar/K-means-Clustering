import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)
df_xy = pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X", y="Y", kind = "scatter")

model1 = KMeans(n_clusters = 4).fit(df_xy)

df_xy.plot(x = "X", y = "Y", c = model1.labels_, kind="scatter", s = 5, cmap = plt.cm.coolwarm)

# Kmeans on crime Data set 
crime1 = pd.read_csv("F:\DS\DS-I\crime_data.csv")

crime1.describe()
crime = crime1.drop([""], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
fd_norm = norm_func(crime1.iloc[:, 1:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 6))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(fd_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(fd_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime1['clust'] = mb # creating a  new column and assigning it to new column 

crime1.head()
fd_norm.head()

crime1 = crime1.iloc[:,[4,0,1,2,3]]
crime1.head()

crime1.iloc[:, 2:4].groupby(crime1.clust).mean()

crime1.to_csv("Kmeans_crime1.csv", encoding = "utf-8")

import os
os.getcwd()
