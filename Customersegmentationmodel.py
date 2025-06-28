#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[3]:


import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[30]:


# lets load the data
df = pd.read_csv('CustomerSegmentation.csv')
df


# In[43]:


df.drop(columns='Unnamed: 0', inplace=True)


# In[45]:


df


# In[47]:


# lets change or fit the gender into intergers 
df.columns


# In[49]:


le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])


# In[51]:


df


# In[53]:


#preview first 10 records
df.head(10)


# In[57]:


#lets begin to create features scaling for clustering
features = ['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender', 'Years as Customer']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[59]:


kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_label = kmeans.fit_predict(X_scaled)
df['Cluster'] = kmeans_label
df


# In[61]:


df['Cluster'].unique()


# In[63]:


# prepare data for supervised learning using random forest classifier
X = df[['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender', 'Years as Customer']]
y = df['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape)
print(y_train.shape)


# In[65]:


model = RandomForestClassifier(random_state=42)


# In[67]:


model


# In[69]:


model.fit(X_train, y_train)


# In[71]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100
print('Predictions:', y_pred)
print('Accuracy Scores:', accuracy)


# In[81]:


newData = pd.DataFrame({
    'CustomerID': [1, 2, 3, 4, 5],
    'Age':[56, 69, 46, 32, 60], 
    'Annual Income (k$)':[47, 67, 69, 52, 48], 
    'Spending Score (1-100)':[81, 50, 61, 13, 12], 
    'Gender':[1, 1, 1, 0, 0], 
    'Years as Customer':[8, 9, 10, 12, 2]})
newprediction = model.predict(newData)
newData['Predicted Cluster'] = newprediction
print(newData)
newprediction


# In[77]:


print(newprediction)


# In[ ]:




