#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
from sklearn import preprocessing, metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("D:/PavanStudy/datasets/CC GENERAL (1).csv")
df.head()


# In[3]:


df.isnull().any()


# In[4]:


df.fillna(df.mean(), inplace=True)
df.isnull().any()


# In[5]:


x = df.drop('CUST_ID', axis = 1)
print(x)


# In[6]:


#Scaling
scaler = StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)


# In[7]:


#Normalizing the data
X_normalized = normalize(X_scaled_array)
X_normalized = pd.DataFrame(X_normalized)


# In[8]:


#Reducing the dimensionality of the Data
pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
principalDf =  pd.DataFrame(data = X_principal, columns = ['principal component1', 'principal component2'])
finalDf = pd.concat([principalDf, df[['TENURE']]], axis = 1)
finalDf.head()


# In[9]:


ac2 = AgglomerativeClustering(n_clusters = 2)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(principalDf['principal component1'], principalDf['principal component2'],
           c = ac2.fit_predict(principalDf), cmap ='rainbow')
plt.show()


# In[10]:


ac3 = AgglomerativeClustering(n_clusters = 3)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(principalDf['principal component1'], principalDf['principal component2'],
           c = ac3.fit_predict(principalDf), cmap ='rainbow')
plt.show()


# In[11]:


ac4 = AgglomerativeClustering(n_clusters = 4)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(principalDf['principal component1'], principalDf['principal component2'],
           c = ac4.fit_predict(principalDf), cmap ='rainbow')
plt.show()


# In[12]:


ac5 = AgglomerativeClustering(n_clusters = 5)
 
# Visualizing the clustering
plt.figure(figsize =(6, 6))
plt.scatter(principalDf['principal component1'], principalDf['principal component2'],
           c = ac5.fit_predict(principalDf), cmap ='rainbow')
plt.show()


# In[13]:


k = [2, 3, 4, 5]
 
# Appending the silhouette scores of the different models to the list
silhouette_scores = []
silhouette_scores.append(
        silhouette_score(principalDf, ac2.fit_predict(principalDf)))
silhouette_scores.append(
        silhouette_score(principalDf, ac3.fit_predict(principalDf)))
silhouette_scores.append(
        silhouette_score(principalDf, ac4.fit_predict(principalDf)))
silhouette_scores.append(
        silhouette_score(principalDf, ac5.fit_predict(principalDf)))


# In[14]:


# Plotting a bar graph to compare the results
plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 10)
plt.ylabel('Silhouette_scores', fontsize = 10)
plt.show()


# In[ ]:




