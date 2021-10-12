#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries and packages 

get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates


# In[ ]:


# Import dataset

df = pd.read_csv('Cereals.csv')
df.set_index('name', inplace=True)


# In[ ]:


# Remove missing values

df=df.dropna()


# In[ ]:


# Check dataset summary 

df.info()


# In[ ]:


df.head()


# In[ ]:


# Drop irrelevant colums in our analysis

df=df.drop(columns=['mfr','type'],axis=1)


# * Apply hierarchical clustering to the normalized data using Euclidean distance.

# In[ ]:


df = df.apply(lambda x: x.astype('float64'))
df.head()


# In[ ]:


# Normalize data

df_norm = df.apply(preprocessing.scale, axis=0)
df_norm


# In[ ]:


# Calculate Euclidean distances

d_norm = pairwise.pairwise_distances(df_norm, metric='euclidean')
pd.DataFrame(d_norm, columns=df_norm.index, index=df_norm.index).head(5)


# * Compare the dendrograms and cluster centroids from single linkage and complete linkage. 
# 

# ### 1. Single Linkage

# In[ ]:


# Single Linkage Dendrogram

Z = linkage(df_norm, method='single')
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(bottom=0.23)
plt.title('Hierarchical Clustering Dendrogram (Single linkage)')
plt.xlabel('Company')
dendrogram(Z, labels=df_norm.index, color_threshold=2.25)
plt.axhline(y=2.25, color='black', linewidth=0.5, linestyle='dashed')
plt.show()


# In[ ]:


# Single Linkage (Minimum distance)

memb = fcluster(linkage(df_norm, 'single'), 7, criterion='maxclust')
memb = pd.Series(memb, index=df_norm.index)
print('\033[1m'+'Single linkage cluster membership:'+'\033[0m') # prefix and suffix for 'bold' print
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# * Obtain cluster centroids for hierarchical clustering by computing the average values of each cluster members

# In[ ]:


# Cluster centroids in single linkage

df_memb=pd.DataFrame(memb)
df_memb=df_memb.rename(columns={0:'clusterlabel'})
df_norm_memb = pd.concat([df_norm,df_memb],axis=1)
centroids = df_norm_memb.groupby(['clusterlabel']).mean()

centroids


# In[ ]:


centroids['clusterlabel'] = ['Cluster {}'.format(i) for i in centroids.index]
plt.figure(figsize=(10,6))
fig.subplots_adjust(right=3)
ax = parallel_coordinates(centroids, class_column='clusterlabel', colormap='Dark2', linewidth=5)
plt.legend(loc='center left', bbox_to_anchor=(0.85, 0.5))
plt.xlim(-0.5,13)


# ### 2. Complete Linkage

# In[ ]:


Z = linkage(df_norm, method='complete')

fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(bottom=0.23)
plt.title('Hierarchical Clustering Dendrogram (Complete linkage)')
plt.xlabel('Company')
dendrogram(Z, labels=df_norm.index, color_threshold=6.5)
plt.axhline(y=6.5, color='black', linewidth=0.5, linestyle='dashed')
plt.show()


# In[ ]:


# Complete Linkage (Maximum distance)

memb = fcluster(linkage(df_norm, 'complete'), 7, criterion='maxclust')
memb = pd.Series(memb, index=df_norm.index)
print('\033[1m'+'Complete linkage cluster membership:'+'\033[0m')
for key, item in memb.groupby(memb):
    print(key, ': ', ', '.join(item.index))


# In[ ]:


# Cluster centroids in complete linkage

df_memb=pd.DataFrame(memb)
df_memb=df_memb.rename(columns={0:'clusterlabel'})
df_norm_memb = pd.concat([df_norm,df_memb],axis=1)
centroids = df_norm_memb.groupby(['clusterlabel']).mean()

centroids


# In[ ]:


centroids['clusterlabel'] = ['Cluster {}'.format(i) for i in centroids.index]
plt.figure(figsize=(10,6))
fig.subplots_adjust(right=3)
ax = parallel_coordinates(centroids, class_column='clusterlabel', colormap='Dark2', linewidth=5)
plt.legend(loc='center left', bbox_to_anchor=(0.85, 0.5))
plt.xlim(-0.5,13)


# ### Comparison between single linkage VS. complete linkage: 
# 
# * Single linkage leans to the left, whereas complete linkage is more evenly distributed, therefore complete linkage is more stable than single linkage.
# 
# 
# ### Number of Clusters = 7
# 
# * using 6.5 (dot line) in the complete linkage for the cutoff to make 7 clusters because it can cut the dendrogram into evenly clusters despite the outlier.
# 
# 
# ### Healthy Cereals Recommendation: 
# Result has shown that cluster 5 has the most healthy cereal brands: 100%_Bran, All-Bran, All-Bran_with_Extra_Fiber. 
# 
# * High in fiber, protein, potassium
# 
# * Low in carbohydrate, calories and fat
