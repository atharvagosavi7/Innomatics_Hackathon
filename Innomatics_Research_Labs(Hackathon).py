#!/usr/bin/env python
# coding: utf-8

# <b>Importing Libraries that we require while performing operations.</b>

# In[43]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data=pd.read_csv("dataframe_.csv")


# In[14]:


data.head(25)


# In[17]:


data.tail(25)


# In[6]:


data


# In[9]:


data.shape


# In[11]:


data.describe()


# In[15]:


print(data.isnull().sum())


# In[16]:


data.isnull()


# In[31]:


data.notnull().sum(axis=0)


# In[21]:


data = data.fillna(data.mean())
data.isnull().sum()


# In[35]:


data1 = data.fillna(data.median())
data1.isnull().sum()


# In[22]:


plt.plot(data['input'])


# In[26]:


plt.plot(data['output'])


# In[27]:


sns.boxplot(data['input'])


# <b>As is evident, input contains no outliers.</b>

# In[28]:


sns.boxplot(data['output'])


# <b>As is evident, output contains outliers.</b>

# In[40]:


data['output'].quantile(0.999)


# In[41]:


data2 = data[data['output'] <= 135]


# In[42]:


data2.describe()


# In[44]:


data.corr()


# In[48]:


sns.regplot(x='input',y='output',data=data)


# <b>As can be seen, input grows with respect to output at a particular level and then declines with respect to output after that level it can seen when both output and input less than zero. When the input exceeds 0, the output increases linearly after that.</b>

# <h1> Linear Regression and PolyRegression</h1>

# In[57]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm


# In[60]:


a=data[['input']]
b=data['output']


# In[61]:


lm.fit(a,b)


# In[62]:


Yhat=lm.predict(a)
Yhat[0:5]   


# In[63]:


lm.intercept_


# In[64]:


lm.coef_


# In[67]:


width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="input", y="output", data=data)
plt.ylim(0,)


# <p><b>The difference between the observed value (y) and the predicted value (Yhat) is called the residual (e). When we look at a regression plot, the residual is the distance from the data point to the fitted regression line.</b></p>

# In[68]:


plt.figure(figsize=(width, height))
sns.regplot(x="input", y="output", data=data)
plt.ylim(0,)


# In[69]:


def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Input')

    plt.show()
    plt.close()


# In[71]:


x = data['input']
y = data['output']


# In[72]:


f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)


# In[73]:


PlotPolly(p, x, y, 'input')


# In[74]:


np.polyfit(x, y, 3)


# <b>We can already see from plotting that this polynomial model performs better than the linear model. This is because the generated polynomial function "hits" more of the data points.</b>

# In[76]:


from sklearn.metrics import r2_score


# In[77]:


r_squared = r2_score(y, p(x))
print('The R-square value polynomial is: ', r_squared)


# In[80]:


lm.fit(a, b)
# Find the R^2
print('The R-square linear regression is: ', lm.score(a, b))


# In[81]:


from sklearn.metrics import mean_squared_error


# In[83]:


mse = mean_squared_error(data['output'], Yhat)
print('The mean square error of price and predicted value is: ', mse)


# In[84]:


mse = mean_squared_error(data['input'], Yhat)
print('The mean square error of price and predicted value is: ', mse)


# In[86]:


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(data['input'],data['output'] , test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# ## Decision Tree

# In[90]:


from sklearn.tree import DecisionTreeRegressor


# In[103]:


dc= DecisionTreeRegressor()
dc


# In[104]:



from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(a, b, test_size= 0.25, random_state=0)


# In[105]:


dc.fit(x_train, y_train)


# In[111]:


dc.score(x_test,y_test)


# <b>The testing is about 61%</b>

# In[107]:


Yhat_dc=dc.predict(a)
Yhat_dc[0:5]


# In[110]:


mse_dc = mean_squared_error(data['output'], Yhat_dc)
print('The mean square error of price and predicted value is: ', mse_dc)


# <h1>Neaural Network</h1>

# In[112]:


from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


# In[114]:


ml=MLPRegressor()
ml


# In[116]:


X, y = make_regression(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
regr.predict(X_test[0:5])
regr.score(X_test, y_test)


# <b>The testing is about 42%</b>

# <ol>For the given dataset I used algorithm: 
#     
#      1.Linear Regression 
#     
#      2.Poly Regression
#     
#      3.Neural Network
#     
#      4.Decision Tree</ol>
#     
#     
#     The best algorithm for this data is Decision Tree that is Maximum with 61%. Higher the accuracy best model is good.

# In[ ]:




