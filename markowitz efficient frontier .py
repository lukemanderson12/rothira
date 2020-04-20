#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


assets = ['VEA','SPY','QQQ','JKE']
pf_data = pd.DataFrame()

for a in assets:
    pf_data[a] = wb.DataReader(a,data_source ='yahoo',start = '2010-1-1')['Adj Close']


# In[54]:


(pf_data / pf_data.iloc[0] * 100).plot(figsize=(10,5))


# In[4]:


log_returns = np.log(pf_data / pf_data.shift(1))


# In[5]:


log_returns.mean() *250


# In[6]:


log_returns.cov()*250


# In[7]:


log_returns.corr()


# In[8]:


num_assets = len(assets)


# In[74]:


pfolio_returns =[]
pfolio_volatilities =[]

for x in range (1000):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    pfolio_returns.append(np.sum(weights * log_returns.mean()) *250)
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T,np.dot(log_returns.cov() * 250,weights))))
pfolio_returns =np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)
    
pfolio_returns, pfolio_volatilities


# In[78]:


portfolios = pd.DataFrame({'Return': pfolio_returns,'Volatility':pfolio_volatilities})
portfolios.to_excel('C:/Python/new_files/RothPort.xlsx')


# In[76]:


portfolios.plot(x='Volatility',y='Return',kind='scatter',figsize=(15,7));
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')


# In[82]:


weights
#in order, VEA, SPY, QQQ, JKE


# In[ ]:




