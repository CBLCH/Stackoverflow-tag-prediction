#!/usr/bin/env python
# coding: utf-8

# In[13]:


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
from sklearn import model_selection
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import preprocessing
import pickle


# In[16]:


pickle_in = open('df_tot.pickle','rb')
df_tot = pickle.load(pickle_in)
pickle_in.close()
pickle_in = open('body_and_title.pickle','rb')
body_and_title = pickle.load(pickle_in)
pickle_in.close()


# In[2]:


list_stop_words = set(nltk.corpus.stopwords.words('english'))    


# In[3]:


def tokenizer_fct(string):
    tokenizer1 = nltk.RegexpTokenizer(r'[a-zA-Z]{2,}')
    lemmatizer = nltk.stem.WordNetLemmatizer() 
    text = tokenizer1.tokenize(string)
    text2 = [lemmatizer.lemmatize(item) for item in text]
    return text2


# In[17]:


t0 = time()
vectorizer = CountVectorizer(
    tokenizer=tokenizer_fct,stop_words=list_stop_words,max_df=0.90 ,min_df=0.05)
tf = vectorizer.fit_transform(body_and_title)
t1 = time()
print(t1-t0)
lda = LatentDirichletAllocation(n_components = 20, random_state=0,n_jobs=-1,max_iter=20)
lda.fit(tf)
t2 = time()
print(t2-t1,t2-t0)


# In[19]:


tf_feature_names = vectorizer.get_feature_names_out()
print(tf_feature_names)


# In[36]:


sns.set_style("white")
n_top_words = 8
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(4, 5, figsize=(36, 18), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Sujet {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
plot_top_words(lda, tf_feature_names, n_top_words, "sujets du modèle LDA")


# In[23]:


topic_distrib = lda.transform(tf)


# In[32]:


topic_distrib[0:1,]


# ### Supervisé compte des tags et de leur représentativité

# In[37]:


compteur_tag = {}
def tag_counter(string):
    tag_tokenizer = nltk.RegexpTokenizer(r'(?<=\<)(.*?)(?=\>)')
    tag_list =tag_tokenizer.tokenize(string)
    for character in tag_list:
        if character in compteur_tag:
            compteur_tag[character]+=1
        else:
            compteur_tag[character]=1   


# In[38]:


df_tot['Tags'].apply(tag_counter)


# In[39]:


compteur_tag


# In[40]:


tag_list = pd.Series(compteur_tag.values(),index=compteur_tag.keys())
most_used_tags = tag_list.nlargest(500)


# In[41]:


compteur_question_couverte = []
def question_couverte(string):
    tag_tokenizer = nltk.RegexpTokenizer(r'(?<=\<)(.*?)(?=\>)')
    x =tag_tokenizer.tokenize(string)
    counter=0    
    for item in x:
        if item in most_used_tags.index:
            counter=1    
    compteur_question_couverte.append(counter)   


# In[52]:


# test_range = [100,150,200,300,400,500]
# question_convered = []
# for range in test_range:
#     most_used_tags = tag_list.nlargest(range)
#     compteur_question_couverte = []
#     df_tot['Tags'].apply(question_couverte)
#     y = sum(compteur_question_couverte)/len(df_tot['Tags'])
#     question_convered.append(y)
# print(question_convered)
fig = sns.scatterplot(test_range,question_convered)
fig.set_xlabel('Nombre de tags')
fig.set_ylabel('Pourcentage de questions couvertes')


# # Supervisé mise en place modèle 1 étiquette

# In[86]:


# On commence par essayer de couvrir 95% de la population des questions. Il faudra donc 150 tags
most_used_tags = tag_list.nlargest(500)
tags = set(most_used_tags.index)
df_tot_sup = df_tot.copy()


# In[53]:


# Enlever les entrées de df_tot_sup qui ne contiennent pas au moins 1 tag parmi les 150 plus fréquents
tag_tokenizer = nltk.RegexpTokenizer(r'(?<=\<)(.*?)(?=\>)')
t0= time()
for count,item in enumerate(df_tot_sup['Tags']):
    tag_pertinent = []
    tag_use_number = []
    tag_liste = tag_tokenizer.tokenize(item)
    for taggs in tag_liste:
        if taggs in tags:
            if tag_pertinent==[]:                
                tag_pertinent =[taggs]
                tag_use_number=[most_used_tags.loc[taggs]]                
            elif tag_use_number[0]<=most_used_tags.loc[taggs]:
                tag_pertinent=[taggs]
                tag_use_number=[most_used_tags.loc[taggs]]  
    if tag_pertinent == []:
        df_tot_sup.drop(count,axis=0,inplace=True)
    else:
        df_tot_sup.Tags.loc[count] = tag_pertinent[0]
t1=time()
print(t1-t0)


# In[54]:


body_title = df_tot_sup['Body']+df_tot_sup['Title']
body_title.name = 'Body_Title'
df_tot_sup_bis = pd.concat([df_tot_sup,body_title],axis=1)


# In[55]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df_tot_sup_bis['Body_Title'],df_tot_sup_bis['Tags'],test_size=0.2,random_state=42)


# In[56]:


t0= time()
tf_vec = TfidfVectorizer(tokenizer=tokenizer_fct,stop_words=list_stop_words,max_df=0.90,min_df=0.05)
tf_vec_fitted = tf_vec.fit(X_train)
bag_of_words = tf_vec_fitted.transform(X_train)
t1 = time()
print(t1-t0)


# In[57]:


t0= time()
bag_of_words_test = tf_vec_fitted.transform(X_test)
t1 = time()
print(t1-t0)


# In[58]:


t0= time()
log_reg = LogisticRegression(C=0.005,multi_class='ovr',n_jobs=-1)
test = log_reg.fit(bag_of_words,y_train)
t1 = time()
print(t1-t0)


# In[59]:


test.score(bag_of_words_test,y_test)


# In[60]:


jaccard_score(test.predict(bag_of_words_test),y_test,average='weighted')


# In[61]:


def tf_idf_log_reg(mindf,maxdf,c_val):
    t0=time()
    tf_vec = TfidfVectorizer(
        tokenizer=tokenizer_fct,stop_words=list_stop_words,max_df=maxdf,min_df=mindf)
    tf_vec_fitted = tf_vec.fit(X_train)
    bag_of_words = tf_vec_fitted.transform(X_train)
    bag_of_words_test = tf_vec_fitted.transform(X_test)
    log_reg = LogisticRegression(C=c_val,multi_class='ovr',n_jobs=-1)
    log_reg_model = log_reg.fit(bag_of_words,y_train)
    jaccard_w = jaccard_score(log_reg_model.predict(bag_of_words_test),y_test,average='weighted')
    jaccard_m = jaccard_score(log_reg_model.predict(bag_of_words_test),y_test,average='micro')
    jaccard_M = jaccard_score(log_reg_model.predict(bag_of_words_test),y_test,average='macro')
    reg_score = log_reg_model.score(bag_of_words_test,y_test)
    f1_sco_w = f1_score(y_test,log_reg_model.predict(bag_of_words_test),                      labels=df_tot_sup['Tags'].unique(),average='weighted')
    liste_score = [jaccard_w,jaccard_m,jaccard_M,reg_score,f1_sco_w,mindf,maxdf,c_val]
    t1=time()
    print(t1-t0)
    return(liste_score)


# In[397]:


liste = []
mindf_liste=[0.01,0.03,0.05]
maxdf_liste=[0.97,0.93,0.9]
c_val_liste=[0.001,0.5,0.01,0.1]
for mindf in mindf_liste:
    for maxdf in maxdf_liste:
        for c_val in c_val_liste:
            x = tf_idf_log_reg(mindf,maxdf,c_val)
            liste.append(x)


# In[398]:


liste


# # Supervisé multi étiquettes

# In[132]:


df_tot_multi = df_tot.copy()
most_used_tags = tag_list.nlargest(150)
tags = set(most_used_tags.index)


# In[133]:


def common_tag(string):
    tag_tokenizer = nltk.RegexpTokenizer(r'(?<=\<)(.*?)(?=\>)')
    tag_retenu = []
    tag_liste = tag_tokenizer.tokenize(string)
    for tagg in tag_liste:
        if tagg in tags:
            tag_retenu.append(tagg)
    return(tag_retenu)


# In[134]:


t0=time()
df_tot_multi['Tags'] = df_tot_multi['Tags'].apply(common_tag)
indexNames = df_tot_multi[df_tot_multi['Tags'].str.len()==0].index
df_tot_multi.drop(indexNames,axis=0,inplace=True)
df_tot_multi.shape
t1=time()
print(t1-t0)


# In[153]:


mlb = preprocessing.MultiLabelBinarizer()
y_true = mlb.fit_transform(df_tot_multi['Tags'])
mlb.classes_


# In[ ]:


X_train_multi, X_test_multi, y_train_multi, y_test_multi = model_selection.train_test_split(
    df_tot_multi['Body_Title_multi'],y_true,test_size=0.2,random_state=42)


# In[ ]:


t0=time()
tf_vec_multi = TfidfVectorizer(
        tokenizer=tokenizer_fct,stop_words=list_stop_words,max_df=0.999,min_df=0.00,ngram_range=(1, 1))
tf_vec_fitted_multi = tf_vec_multi.fit(X_train_multi)
bag_of_words_multi = tf_vec_fitted_multi.transform(X_train_multi)
bag_of_words_multi_test = tf_vec_fitted_multi.transform(X_test_multi)
t1=time()
print(t1-t0)


# In[ ]:


jaccard_score(y_test_multi,clf_multi.predict(bag_of_words_multi_test),average='weighted')


# In[ ]:


t0=time()
log_reg_multi = LogisticRegression(max_iter=3000,tol=1e-6)
clf_multi = OneVsRestClassifier(log_reg_multi)
clf_multi.fit(bag_of_words_multi,y_train_multi)
test = clf_multi.score(bag_of_words_multi_test,y_test_multi)
t1=time()
print(t1-t0)
print(test)
# log_reg_multi.fit(bag_of_words_multi,y_true)


# In[18]:


def tf_idf_log_reg_multi(mini,maxi,c_val,ngram):
    t0=time()
    tf_vec_multi = TfidfVectorizer(
        tokenizer=tokenizer_fct,stop_words=list_stop_words,max_df=maxi,min_df=mini,ngram_range=ngram)
    tf_vec_fitted_multi = tf_vec_multi.fit(X_train_multi)
    bag_of_words_multi = tf_vec_fitted_multi.transform(X_train_multi)
    bag_of_words_multi_test = tf_vec_fitted_multi.transform(X_test_multi)
    log_reg_multi = LogisticRegression(C=c_val,max_iter=2000,tol=1e-4,solver='saga')
    clf_multi = OneVsRestClassifier(log_reg_multi,n_jobs=-1)
    clf_multi.fit(bag_of_words_multi,y_train_multi)
    jaccard_w = jaccard_score(y_test_multi,clf_multi.predict(bag_of_words_multi_test),average='weighted')
    jaccard_m = jaccard_score(y_test_multi,clf_multi.predict(bag_of_words_multi_test),average='micro')
    jaccard_M = jaccard_score(y_test_multi,clf_multi.predict(bag_of_words_multi_test),average='macro')
    reg_score = clf_multi.score(bag_of_words_multi_test,y_test_multi)
#     f1_sco_w = f1_score(y_test_multi,clf_multi.predict(bag_of_words_multi_test),\
#                       labels=df_tot_multi['Tags'].unique(),average='weighted')
    liste_score = [jaccard_w,jaccard_m,jaccard_M,reg_score,mini,maxi,c_val,ngram,loga]
    t1=time()
    print(t1-t0)
    return(liste_score)


# In[ ]:


liste_multi = []
mini_liste=[0.00,0.02,0.05]
maxi_liste=[0.8,0.9,0.95]
c_val_liste=[0.95,0.9,0.5,0.1]
ngram_liste=[(1,1)]
loga='l1'
for mini in mini_liste:
    for maxi in maxi_liste:
        for c_val in c_val_liste:
            for ngram in ngram_liste:                
                x = tf_idf_log_reg_multi(mini,maxi,c_val,ngram)
                liste_multi.append(x)
                pickle_liste_multi = open('liste_multi.pickle','wb')
                pickle.dump(liste_multi,pickle_liste_multi)
                pickle_liste_multi.close()

