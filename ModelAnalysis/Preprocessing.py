#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
#from mpl_toolkits.mpot3d import Axes3D
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn import metrics
from sklearn import metrics as mr
from sklearn.feature_selection import RFECV
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from mpl_toolkits.mplot3d import Axes3D


# In[32]:


dataset = pd.read_csv('ENB2012_data.csv')
dataset.info()
#print(dataset)


# In[33]:


#look at  data
peek = dataset.head(20)
print(peek)


# In[34]:


#dimension of the data
shape = dataset.shape
print(shape)


# In[35]:


# data types
types = dataset.dtypes
print(types)


# In[45]:


#change the type and describe
dataset['X6'] = dataset['X6'].astype('float')
dataset['X8'] = dataset['X8'].astype('float')
#dataset.info()
dataset.describe()


# In[9]:


#replace null
dataset = dataset.replace(to_replace='?',value=np.nan)
dataset = dataset.dropna(how='any')


# In[19]:


#split data in training and test set
train = dataset.sample(frac=0.8, random_state=200)
test = dataset.drop(train.index)
#print(test.shape)


# In[36]:


# descriptive statistics
description = train.describe()
des = description.round(decimals=3)
des.to_csv('description.csv', index=True, header=True)
#print(description)


# In[37]:


correlations = train.corr(method='spearman')
corr = correlations.round(decimals=3)
corr.to_csv('correlations.csv', index=True, header=True)


# In[38]:


train.hist('X1')
train.hist('X2')
train.hist('X3')
train.hist('X4')
train.hist('X5')
train.hist('X6')
train.hist('X7')
train.hist('X8')
train.hist('Y1')
train.hist('Y2')


# In[44]:


#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X1'], train['Y1'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X1')
ax.set_ylabel('Y1')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X2'], train['Y1'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X2')
ax.set_ylabel('Y1')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X3'], train['Y1'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X3')
ax.set_ylabel('Y1')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X4'], train['Y1'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X4')
ax.set_ylabel('Y1')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X5'], train['Y1'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X5')
ax.set_ylabel('Y1')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X6'], train['Y1'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X6')
ax.set_ylabel('Y1')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X7'], train['Y1'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X7')
ax.set_ylabel('Y1')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X8'], train['Y1'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X8')
ax.set_ylabel('Y1')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X1'], train['Y2'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X1')
ax.set_ylabel('Y2')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X2'], train['Y2'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X2')
ax.set_ylabel('Y2')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X3'], train['Y2'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X3')
ax.set_ylabel('Y2')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X4'], train['Y2'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X4')
ax.set_ylabel('Y2')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X5'], train['Y2'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X5')
ax.set_ylabel('Y2')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X6'], train['Y2'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X6')
ax.set_ylabel('Y2')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X7'], train['Y2'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X7')
ax.set_ylabel('Y2')
plt.show()

#get a figure object and an axix object to manipulate
fig, ax = plt.subplots()

#creat a scatter plot and lable the axes
ax.scatter(train['X8'], train['Y2'], color = 'blue', alpha = .8, s = 140, marker = 'o')
ax.set_xlabel('X8')
ax.set_ylabel('Y2')
plt.show()


# In[15]:


#Spearman rank correlation coefficient
# Correlation analysis
correlations = train.corr(method='spearman')
corr = correlations.round(decimals=3)
corr.to_csv('correlations.csv', index=True, header=True)
#print(correlations)
df = train[['X1', 'Y1']]
print(df.corr('spearman'))
df = train[['X2', 'Y1']]
print(df.corr('spearman'))
df = train[['X3', 'Y1']]
print(df.corr('spearman'))
df = train[['X4', 'Y1']]
print(df.corr('spearman'))
df = train[['X5', 'Y1']]
print(df.corr('spearman'))
df = train[['X6', 'Y1']]
print(df.corr('spearman'))
df = train[['X7', 'Y1']]
print(df.corr('spearman'))
df = train[['X8', 'Y1']]
print(df.corr('spearman'))


# In[16]:


#Spearman rank correlation coefficient
df = train[['X1', 'Y2']]
print(df.corr('spearman'))
df = train[['X2', 'Y2']]
print(df.corr('spearman'))
df = train[['X3', 'Y2']]
print(df.corr('spearman'))
df = train[['X4', 'Y2']]
print(df.corr('spearman'))
df = train[['X5', 'Y2']]
print(df.corr('spearman'))
df = train[['X6', 'Y2']]
print(df.corr('spearman'))
df = train[['X7', 'Y2']]
print(df.corr('spearman'))
df = train[['X8', 'Y2']]
print(df.corr('spearman'))


# In[ ]:




