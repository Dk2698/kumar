#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


data = pd.read_csv("Housing_Modified_prepared.csv")
data.head()


# In[5]:


# convert text to number using label binarizer
import sklearn.preprocessing as pp
lb =pp.LabelBinarizer()
data.fullbase = lb.fit_transform(data.fullbase)
data.gashw = lb.fit_transform(data.gashw)
data.airco = lb.fit_transform(data.airco)


data.head(3)


# In[6]:


Y = data["price"]
independent_variables = data.columns
independent_variables =independent_variables.delete(0)
X = data[independent_variables]


# In[8]:


# Train the model using Linear regresssion equation
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS

lr = LinearRegression()
lr.fit(X,Y)


# In[9]:


# Train themodel using Ordinary least Square regresssion
from statsmodels.api import OLS
ols =OLS(X,Y).fit()


# In[10]:


# predict using the Linear regression model
Ypred = lr.predict(X)
Ypred


# In[12]:


# calculate R2 score for  linear regression model using
# sum of total(SST) and sum of Squared residual(SSR)
# calculate sum of squared Total(sum of(y- ymean)^2)
Ymean=Y.mean()
print("mean of actual dependent variables",Ymean)

# calculate square of y-ymean using numpy  library
import numpy as np
squared_total =np.square(Y-Ymean)
print("suare total is",squared_total)
sst = squared_total.sum()
print("sum of squared total",sst)


# In[13]:


# calcaulate the sum of squared residual (SSR)
squared_residual = np.square(Ypred - Ymean)
ssr = squared_residual.sum()
print("Sum of square residual",ssr)


# In[14]:


# calcualte the value of R2 score (Rsquared)
r2score = ssr / sst
print("R2 score is",r2score)


# In[16]:


# calcuate root mean squared error(RMSE)
error =Y - Ypred
print("Errors",error)
# take a suare of error
square_error = np.square(error)
print("Squared errors",square_error)


# In[18]:


# calcuate mean squard errors
sse=square_error.sum()
mse =sse /len(Y)
print("Mean suared error",mse)
print("number of element",len(Y))
rmse = np.sqrt(mse)
print("root mean sqared roors",rmse)


# In[19]:


# calculate mean absolute error
#calculate absolute error
absolute_error = abs(Y- Ypred)
sae = absolute_error.sum()
mae = sae /len(Y)
print("mean absolute error", mae)


# In[20]:


# calcuate R2score Rmse and MAE using sklearn
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
r2score=r2_score(Y,Ypred)
print("R2 Score",r2score)


# ## # Objective: 
# # Removeal of multi-col linearity from housing Dataset by calcualting variance influation factor
# 

# In[21]:


# fit the ordinnary last suared regression model
import statsmodels.api as sm
model= sm.OLS(Y,X)
# train the model
model= model.fit()
# check tje model summary
model.summary()


# In[45]:


user_inut ={}
for var in independent_variables:
    temp = input("enter" +var +":")
    user_inut[var] =temp
user_df=pd.DataFrame(data=user_inut,index=[0],columns=independent_variables)
#price = model.predict(user_df)   
import sklearn.linear_model as lm
lr= lm.LinearRegression()
lr.fit(X,Y)
price = lr.predict(user_df)
print(" House price  is USD",int(price[0]))


# In[44]:


user_df


# In[43]:


model.predict(user_df)


# In[34]:


# calculate variance inflation factor
from  statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(len(independent_variables)):
        vif_list = [variance_inflation_factor(data[independent_variables].values,index) for index in range(len(independent_variables))]
        mvif =max(vif_list)
        print("max VIF value is",mvif)
        drop_index = vif_list.index(mvif)
        print("for the independent variables",independent_variables[drop_index])
        if mvif >5:
            print("deleting",independent_variables[drop_index])
            independent_variables = independent_variables.delete(drop_index)
print("final independet variable",independent_variables)


# In[35]:


# calculate variance inflation factor
from  statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(len(independent_variables)):
        vif_list = [variance_inflation_factor(data[independent_variables].values,index) for index in range(len(independent_variables))]
        mvif =max(vif_list)
        print("max VIF value is",mvif)
        drop_index = vif_list.index(mvif)
        print("for the independent variables",independent_variables[drop_index])
        if mvif >10:
            print("deleting",independent_variables[drop_index])
            independent_variables = independent_variables.delete(drop_index)
print("final independet variable",independent_variables)


# In[37]:


X= data[independent_variables]
Y= data["price"]
model= sm.OLS(Y,X)
model = model.fit()
model.summary()


# In[ ]:




