#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[92]:


dataset = pd.read_csv('ENB2012_data.csv')


# In[93]:


# Split data in training and test set
train = dataset.sample(frac=0.8, random_state=200)
test = dataset.drop(train.index)


# In[94]:


print(train['Y1'].describe())
print(train['Y2'].describe())


# In[95]:


# data preprocessing
x_Train = train[['X1', 'X2', 'X3', 'X5', 'X7']].values
y_Train = train[['Y1', 'Y2']].values
x_Test = test[['X1', 'X2', 'X3','X5', 'X7']].values
y_Test = test[['Y1', 'Y2']].values
y_Train1 = train[['Y1']].values
y_Train2 = train[['Y2']].values
y_Test1 = test[['Y1']].values
y_Test2 = test[['Y2']].values
#print(x_Train.shape)
#print(y_Train1.shape)
#print(y_Test1.shape)


# In[96]:


# Linear Regression Y1
plt.figure()
model = LinearRegression()
model.fit(x_Train, y_Train1)
y_hat = model.predict(x_Train)
y_hatTest = model.predict(x_Test)
print('Linear Regression Results (Y1)')
print(mean_squared_error(y_Train1, y_hat))
print(mean_squared_error(y_Test1, y_hatTest))
print(mean_absolute_error(y_Train1, y_hat))
print(mean_absolute_error(y_Test1, y_hatTest))
print(r2_score(y_Train1,y_hat))
print(r2_score(y_Test1,y_hatTest))


# In[97]:


# Linear Regression Y2
plt.figure()
model = LinearRegression()
model.fit(x_Train, y_Train2)
y_hat = model.predict(x_Train)
y_hatTest = model.predict(x_Test)
print('Linear Regression Results (Y2)')
print(mean_squared_error(y_Train2, y_hat))
print(mean_squared_error(y_Test2, y_hatTest))
print(mean_absolute_error(y_Train2, y_hat))
print(mean_absolute_error(y_Test2, y_hatTest))
print(r2_score(y_Train2,y_hat))
print(r2_score(y_Test2,y_hatTest))


# In[98]:


# RandomForest Regressor y1
rfrmodel = RFECV(RandomForestRegressor(), cv=3, scoring='neg_mean_squared_error', step=1)
rfrmodel.fit(x_Train,y_Train1)
x_rfr = rfrmodel.transform(x_Train)
print(x_Train.shape)
print(x_rfr.shape)
print(rfrmodel.support_)
y_hatRFR = rfrmodel.predict(x_Train)
y_hatRFRTest = rfrmodel.predict(x_Test)
print('Random Forrest Regression Results (Y1)')
print(mean_squared_error(y_Train1, y_hatRFR))
print(mean_squared_error(y_Test1, y_hatRFRTest))
print(mean_absolute_error(y_Train1, y_hatRFR))
print(mean_absolute_error(y_Test1, y_hatRFRTest))
print(r2_score(y_Train1, y_hatRFR))
print(r2_score(y_Test1, y_hatRFRTest))


# In[99]:


# RandomForest Regressor y2
rfrmodel = RFECV(RandomForestRegressor(), cv=3, scoring='neg_mean_squared_error', step=1)
rfrmodel.fit(x_Train,y_Train2)
x_rfr = rfrmodel.transform(x_Train)
print(x_Train.shape)
print(x_rfr.shape)
print(rfrmodel.support_)
y_hatRFR = rfrmodel.predict(x_Train)
y_hatRFRTest = rfrmodel.predict(x_Test)
print('Random Forrest Regression Results (Y2)')
print(mean_squared_error(y_Train2, y_hatRFR))
print(mean_squared_error(y_Test2, y_hatRFRTest))
print(mean_absolute_error(y_Train2, y_hatRFR))
print(mean_absolute_error(y_Test2, y_hatRFRTest))
print(r2_score(y_Train2, y_hatRFR))
print(r2_score(y_Test2, y_hatRFRTest))


# In[ ]:




