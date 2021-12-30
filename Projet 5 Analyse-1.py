#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time 
import sys
import csv
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from bs4 import BeautifulSoup
from sklearn import preprocessing
import pickle


# In[2]:


df0 = pd.read_csv('QueryResults.csv')
df1 = pd.read_csv('QueryResults (1).csv')
df2 = pd.read_csv('QueryResults (2).csv')
df3 = pd.read_csv('QueryResults (3).csv')
df4 = pd.read_csv('QueryResults (4).csv')
df5 = pd.read_csv('QueryResults (5).csv')
df6 = pd.read_csv('QueryResults (6).csv')
df7 = pd.read_csv('QueryResults (7).csv')
df8 = pd.read_csv('QueryResults (8).csv')
df9 = pd.read_csv('QueryResults (9).csv')
df10 = pd.read_csv('QueryResults (10).csv')
Liste = [df0,df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]


# In[3]:


df_tot = pd.concat(Liste,axis=0,ignore_index=True)
print(df_tot.shape)
print(df_tot.columns)


# In[4]:


x = set(df_tot.columns)-set(['Id','AcceptedAnswerId','Score','ViewCount','Body','Title','Tags'])
df_tot.drop(x,axis=1,inplace=True)


# In[5]:


def url_remover(text):
    soup = BeautifulSoup(text)
    x = soup.find_all('a')
    [item.decompose() for item in x]
    cleared_text = soup.get_text()
    return(cleared_text)


# In[6]:


compteur_mots={}
def word_counter(string):
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    word_list =tokenizer.tokenize(string)
    for character in word_list:
        if character in compteur_mots:
            compteur_mots[character]+=1
        else:
            compteur_mots[character]=1    


# In[7]:


df_tot['Body']=df_tot['Body'].apply(url_remover)
body_and_title = test1=df_tot['Body']+df_tot['Title']


# In[8]:


body_and_title.apply(word_counter)


# In[9]:


word_list = pd.Series(compteur_mots.values(),index=compteur_mots.keys())


# In[10]:


compteur_mots


# In[11]:


list_stop_words = set(nltk.corpus.stopwords.words('english'))
for word in word_list.index:
    if word.lower() in list_stop_words:
        word_list.drop(word,inplace=True)       


# In[12]:


word_list.nlargest(60)


# In[16]:


compteur_mots_bis={}
def word_counter_bis(string):
    tokenizer = nltk.RegexpTokenizer(r'[a-zA-Z]{2,}')
    word_list =tokenizer.tokenize(string)
    for character in word_list:
        if character in compteur_mots_bis:
            compteur_mots_bis[character]+=1
        else:
            compteur_mots_bis[character]=1  


# In[17]:


body_and_title.apply(word_counter_bis)


# In[18]:


word_list_bis = pd.Series(compteur_mots_bis.values(),index=compteur_mots_bis.keys())


# In[19]:


for word in word_list_bis.index:
    if word.lower() in list_stop_words:
        word_list_bis.drop(word,inplace=True)


# In[20]:


word_list_bis.nlargest(60)


# In[26]:


pickle_tok_fct = open('df_tot.pickle','wb')
pickle.dump(df_tot,pickle_tok_fct)
pickle_tok_fct.close()
pickle_tok_fct = open('body_and_title.pickle','wb')
pickle.dump(body_and_title,pickle_tok_fct)
pickle_tok_fct.close()

