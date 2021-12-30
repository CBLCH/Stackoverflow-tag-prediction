#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import streamlit as st
import pickle
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[2]:


def url_remover(text):
    soup = BeautifulSoup(text)
    x = soup.find_all('a')
    [item.decompose() for item in x]
    cleared_text = soup.get_text()
    return(cleared_text)


# In[3]:


def tokenizer_fct(string):
    tokenizer1 = nltk.RegexpTokenizer(r'[a-zA-Z]{2,}')
    lemmatizer = nltk.stem.WordNetLemmatizer() 
    text = tokenizer1.tokenize(string)
    text2 = [lemmatizer.lemmatize(item) for item in text]
    return text2


# In[4]:


# x=os.getcwd()


# In[4]:


# os.chdir('C:\\Users\\SANDRA\\formation\\Projet 5')


# In[5]:


pickle_in = open('df_tot_multi.pickle','rb')
df_tot_multi = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('mlb.pickle','rb')
mlb = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('classifier.pickle','rb')
classifier = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('tf_vec_fitted_multi.pickle','rb')
tf_vec_fitted_multi = pickle.load(pickle_in)
pickle_in.close()


# In[7]:


txt =st.text_area('Text you wish to tag')


# In[10]:


txt = url_remover(txt)


# In[11]:


text = tf_vec_fitted_multi.transform([txt])


# In[12]:


x=classifier.predict_proba(text)


# In[13]:


x1 = pd.Series(x[0])
x1 = x1.transpose()
# label = x1.nlargest(n).index


# In[14]:


z=x1[x1>0.2].index


# In[16]:


u=mlb.classes_[z]


# In[27]:


st.write('your labels are',u)


# In[21]:


# df_tot_multi['Body_Title_multi'][0]

