#!/usr/bin/env python
# coding: utf-8

# In[23]:


from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix
import pandas as pd
from sklearn.model_selection import train_test_split

dados_iris = load_iris()


# In[4]:


dados_iris.keys()


# In[18]:


print(dados_iris["data"][0:5])


# In[21]:


print(dados_iris["target"][0:5])


# In[19]:


print(dados_iris["target_names"][0:5])


# In[20]:


print(dados_iris["feature_names"][0:5])


# In[5]:


print(dados_iris['DESCR'])


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(dados_iris['data'], dados_iris['target'], random_state=0)


# In[25]:


print(" Treino: {}".format(X_train.shape))
print("Teste: {}".format(X_test.shape))


# In[28]:


iris_dataframe = pd.DataFrame(X_train, columns = dados_iris.feature_names)
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker = 'o', hist_kwds={'bins':20}, s=60, alpha =.8)


# In[32]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)


# In[33]:


print("Treino: {}".format(knn.score(X_train, y_train)))
print("Teste: {}". format(knn.score(X_test, y_test)))

