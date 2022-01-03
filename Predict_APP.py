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
import requests
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
classifier_pick = st.file_uploader('Upload the classifier file here. You can download my version here https://drive.google.com/file/d/15uJpphNm0zStOYrIFvazGpnZSupDWwPs/view')
def url_remover(text):
    soup = BeautifulSoup(text)
    x = soup.find_all('a')
    [item.decompose() for item in x]
    cleared_text = soup.get_text()
    return(cleared_text)
def tokenizer_fct(string):
    tokenizer1 = nltk.RegexpTokenizer(r'[a-zA-Z]{2,}')
    lemmatizer = nltk.stem.WordNetLemmatizer() 
    text = tokenizer1.tokenize(string)
    text2 = [lemmatizer.lemmatize(item) for item in text]
    return text2
txt =st.text_area('Text you wish to tag')
pickle_in = open('mlb_100.pickle','rb')
mlb = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('classifier_100.pickle','rb')
classifier = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('tf_vec_fitted_multi.pickle','rb')
tf_vec_fitted_multi = pickle.load(pickle_in)
pickle_in.close()
if len(txt)!=0:
    txt = url_remover(txt)
    text = tf_vec_fitted_multi.transform([txt])
    x=classifier.predict_proba(text)
    x1 = pd.Series(x[0])
    x1 = x1.transpose()
    z=x1[x1>0.2].index
    u=mlb.classes_[z]
    st.write('your labels are',u)

