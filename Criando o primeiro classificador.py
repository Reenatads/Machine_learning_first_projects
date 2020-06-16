#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
plt.figure(figsize=(4,4))
plt.subplot(111)
X1,Y1 = make_classification(n_samples=150,n_features=2, n_redundant=0, n_informative=2, 
                           n_clusters_per_class=1, random_state=3, class_sep= 1.8, n_classes=3)
plt.scatter(X1[:,0], X1[:,1], marker="o", c=Y1, s=25, edgecolor="k")
plt.show()


# In[5]:


from sklearn.model_selection import train_test_split #Função que divide dados de treino e de teste.
from sklearn.neighbors import KNeighborsClassifier #K-Nearest Neighbors
import numpy as np

knn = KNeighborsClassifier(n_neighbors = 1)

X_train, X_test, y_train, y_test = train_test_split(X1, Y1, random_state = 1)

knn.fit(X_train,y_train)

print("Treino = {} \n".format(knn.score(X_train, y_train)))
print("Teste = {} \n".format(knn.score(X_test,y_test)))


# In[13]:


from matplotlib.colors import ListedColormap
x_min, x_max = X1[:,0].min() - 1, X1[:,0].max() + 1
y_min, y_max = X1[:,1].min() - 1, X1[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                    np.arange(y_min, y_max, 0.1))

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
plt.scatter(X1[:,0], X1[:,1], c=Y1, cmap=cmap_bold,
           edgecolor="k", s=20)

plt.show()


# In[ ]:




