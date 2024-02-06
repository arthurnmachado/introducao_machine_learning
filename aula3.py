#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados = pd.read_csv(uri)
dados.head()


# In[ ]:


renomear = {
    "expected_hours": "horas_esperadas",
    "price": "preco",
    "unfinished": "nao_terminado"
}

dados = dados.rename(columns=renomear)
dados.head()


# In[ ]:


trocar = {
    0: 1,
    1: 0
}

dados['finalizado'] = dados.nao_terminado.map(trocar)
dados.tail()


# In[ ]:


import seaborn as sns

sns.scatterplot(x="horas_esperadas", y="preco", hue="finalizado", data=dados)


# In[ ]:


sns.relplot(x="horas_esperadas", y="preco", hue="finalizado", col="finalizado", data=dados)


# In[ ]:


x = dados[["horas_esperadas", "preco"]]
y = dados["finalizado"]


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 5

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,
                                                         random_state = SEED, test_size = 0.25,
                                                         stratify = y)
print(f"Treinaremos com {len(treino_x)} elementos e testaremos com {len(teste_x)} elementos")

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"A acurácia foi {acuracia:.2f} %")


# In[ ]:


import numpy as np

previsoes_de_base = np.ones(540)
acuracia = accuracy_score(teste_y, previsoes_de_base) * 100
print(f"A acurácia do algoritmo de baseline foi {acuracia:.2f} %")


# In[ ]:


sns.scatterplot(x="horas_esperadas", y="preco", hue=teste_y, data=teste_x)


# In[ ]:


x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()
print(x_min, x_max, y_min, y_max)


# In[ ]:


pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / 100)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / 100)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]
pontos


# In[ ]:


Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)
Z.shape


# In[ ]:


import matplotlib.pyplot as plt

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=2)

# DECISION BOUNDARY


# In[ ]:




