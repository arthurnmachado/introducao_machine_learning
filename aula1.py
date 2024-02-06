#!/usr/bin/env python
# coding: utf-8

# In[1]:


# features (1 sim, 0 não)
# pelo longo?
# perna curta?
# faz au au?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# 1 -> porco, 0 -> cachorro
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1, 1, 1, 0, 0, 0] # labels


# In[2]:


from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(treino_x, treino_y)


# In[3]:


animal_misterioso = [1, 1, 1]
model.predict([animal_misterioso])


# In[4]:


misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]
teste_x = [misterio1, misterio2, misterio3]
teste_y = [0, 1, 1] # respostas corretas


# In[5]:


previsoes = model.predict(teste_x)
previsoes


# In[6]:


previsoes == teste_y


# In[7]:


corretos = (previsoes == teste_y).sum()
total = len(teste_x)
taxa_acerto = corretos / total
print(f"Taxa de acerto: {taxa_acerto * 100:.2f} %")


# In[8]:


from sklearn.metrics import accuracy_score

taxa_acerto = accuracy_score(teste_y, previsoes)
print(f"Taxa de acerto: {taxa_acerto * 100:.2f} %")


# In[8]:




