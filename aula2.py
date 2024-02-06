#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri)
dados.head()


# In[8]:


mapa = {
    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou"
}
dados = dados.rename(columns=mapa)


# In[10]:


x = dados[["principal", "como_funciona", "contato"]]
x.head()


# In[11]:


y = dados["comprou"]
y.head()


# In[12]:


dados.shape


# In[18]:


treino_x = x[:75]
treino_y = y[:75]
teste_x = x[75:]
teste_y = y[75:]

print(f"Treinaremos com {len(treino_x)} elementos e testaremos com {len(teste_x)} elementos")


# In[20]:


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)
acuracia = accuracy_score(teste_y, previsoes) * 100

print(f"Taxa de acerto: {acuracia:.2f} %")


# # Usando a biblioteca para separar treino e teste

# In[28]:


from sklearn.model_selection import train_test_split

SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, random_state=SEED)
print(f"Treinaremos com {len(treino_x)} elementos e testaremos com {len(teste_x)} elementos")

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)
acuracia = accuracy_score(teste_y, previsoes) * 100

print(f"Taxa de acerto: {acuracia:.2f} %")


# In[30]:


treino_y.value_counts()


# In[32]:


teste_y.value_counts()


# In[34]:


from sklearn.model_selection import train_test_split

SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, random_state=SEED,
                                                        stratify=y)
print(f"Treinaremos com {len(treino_x)} elementos e testaremos com {len(teste_x)} elementos")

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)
acuracia = accuracy_score(teste_y, previsoes) * 100

print(f"Taxa de acerto: {acuracia:.2f} %")


# In[35]:


treino_y.value_counts()


# In[36]:


teste_y.value_counts()


# In[ ]:




