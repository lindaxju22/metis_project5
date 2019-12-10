#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:56:44 2019

@author: lindaxju
"""

#%%
import sys
sys.setrecursionlimit(100000)

import pickle

import pandas as pd

from datetime import datetime

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

from collections import Counter
#%%
###############################################################################
##############################Helper Functions#################################
###############################################################################
#%%
###################################Vectorize###################################
#%%
def get_doc_word_vectorizer(vectorizer,ngram_range,list_docs):
            
    """
    Input is name of vectorizer, range of ngrams, and list of docs
    Returns vectorizer object and doc word object
    """
    
    if vectorizer == 'cv':
        vec = CountVectorizer(ngram_range=ngram_range,stop_words='english')
    elif vectorizer == 'tfidf':
        vec = TfidfVectorizer(ngram_range=ngram_range,stop_words='english')
    doc_word = vec.fit_transform(list_docs)
    
    return vec, doc_word
#%%
def get_doc_word_df(doc_word,vectorizer,list_docs):
            
    """
    Input is doc word object, vectorizer object, and list of docs
    Returns dataframe of doc word
    """
    
    doc_word_df = pd.DataFrame(doc_word.toarray(),index=list_docs,
                               columns=vectorizer.get_feature_names())
    return doc_word_df
#%%
##########################Dim Reduction w/ LSA and NMF#########################
#%%
def get_list_components(num_components):
            
    """
    Input is number of components
    Returns list of component names
    """
    
    list_components = []
    for num in range(num_components):
        list_components.append('c'+str(num+1).zfill(2))
        
    return list_components
#%%
def get_dim_red(svd,doc_word,num_components):
            
    """
    Input is svd method, doc word object, and number of components
    Returns dimensionality reduction object and doc topic object
    """
    
    if svd == 'lsa':
        dim_red = TruncatedSVD(num_components,random_state=42)
    elif svd == 'nmf':
        dim_red = NMF(num_components,random_state=42)
    doc_topic = dim_red.fit_transform(doc_word)
    
    return dim_red, doc_topic
#%%
def display_topics(model,feature_names,num_components,num_top_words,topic_names=None):
            
    """
    Input is dimensionality reduction object, vectorizer column names, and number words to show
    Prints the top # of words for each topic and returns a dataframe
    """
    
    df = pd.DataFrame(columns=['cluster', 'topic_words'])
    list_components = get_list_components(num_components)
    
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix+1)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))
    
        df.loc[ix] = [list_components[ix]] + [", ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]])]
        
    return df
#%%
def get_doc_topic_max_df(doc_topic,cols,rnd=3):

    """
    Input is doc topic object and names of columns
    Returns topics and topic max dataframe of transcript
    """
    
    doc_topic_df = pd.DataFrame(doc_topic.round(rnd),columns=cols)
    # below code is for each paragraph, but actually want for each transcript
#    doc_topic_df['topic_max'] = doc_topic_df.max(axis=1)
#    doc_topic_df['cluster'] = doc_topic_df.idxmax(axis=1)
    
    return doc_topic_df
#%%
###################################Get Topics##################################
#%%
def get_topics(df_company,num_components):
    
    """
    Input is dataframe of transcript name, original paragraph, and cleaned paragraph (presentation only) and number of topics
    Returns vectorizer, dimensionality reduction, and final company dataframe with topics
    """
  
    list_components = get_list_components(num_components)
    
    # TF-IDF
    vectorizer_tfidf_presentation,doc_word_tfidf_presentation = get_doc_word_vectorizer(vectorizer='tfidf',ngram_range=(1,2),list_docs=df_company.content_cleaned)
    
    # NMF
    nmf_tfidf_presentation,doc_topic_nmf_tfidf_presentation = get_dim_red(svd='nmf',doc_word=doc_word_tfidf_presentation,num_components=num_components)
    
    
    doc_topic_df = get_doc_topic_max_df(doc_topic=doc_topic_nmf_tfidf_presentation,cols=list_components,rnd=3)
    df_company = pd.concat([df_company,doc_topic_df],axis=1)
    
    return vectorizer_tfidf_presentation,nmf_tfidf_presentation,df_company
################################Sort in Order##################################
#%%
def sort_company_df(df_to_be_sorted):
                
    """
    Input is dataframe to be sorted with transcript names including dates
    Returns dataframe sorted by dates
    """
    
    transcript_dates = []
    for transcript_name in df_to_be_sorted.transcript_name:
        transcript_dates.append(datetime.strptime(transcript_name[1], '%m/%d/%y'))
    
    df_sorted = df_to_be_sorted.copy()
    df_sorted.insert(loc=0,column='date',value=transcript_dates)
    
    df_sorted.sort_values(by=['date'],inplace=True)
    df_sorted.reset_index(drop=True,inplace=True)
    
    return df_sorted
#%%
###############################################################################
############################Get Topics Over Time###############################
###############################################################################
#%%
##########################Import Cleaned Transcripts###########################
#%%
filename = 'data/' + 'company_df_2019-12-07-16-01-29.pickle'
with open(filename,'rb') as read_file:
    company_df = pickle.load(read_file)
#%%
########################Get Company Topics Across Time#########################
#%%
# set number of topics and get topics per paragraph
num_components = 30
vectorizer_tfidf_presentation,nmf_tfidf_presentation,company_topics_df = get_topics(company_df,num_components)
topics_df = display_topics(nmf_tfidf_presentation,vectorizer_tfidf_presentation.get_feature_names(),num_components,5)
company_topics_df.sample(5)
#%%
company_topics_df.shape
#%%
#from datetime import datetime
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'company_topics_df'+'_'+timestamp
#company_topics_df.to_csv(r'data/'+filename+'.csv')
#%%
# get topic per transcript by aggregating by cluster
# aggregate by cluster
company_topics_avg_df = company_topics_df.groupby(['transcript_name']).mean().reset_index()
company_topics_avg_df_temp = company_topics_avg_df.drop('transcript_name',axis=1)
# assign topic by max
company_topics_avg_df['topic_max'] = company_topics_avg_df_temp.max(axis=1)
company_topics_avg_df['cluster'] = company_topics_avg_df_temp.idxmax(axis=1)
company_topics_avg_df = pd.merge(company_topics_avg_df,topics_df,on='cluster',how='left')
company_topics_avg_df.shape
#%%
topics_df
#%%
sorted(Counter(company_topics_avg_df.cluster).items())
#%%
company_topics_ordered_df = sort_company_df(company_topics_avg_df)
company_topics_ordered_df
#%%
#from datetime import datetime
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'company_topics_ordered_df'+'_'+timestamp
#company_topics_ordered_df.to_csv(r'data/'+filename+'.csv')
#%%
#import sys
#sys.setrecursionlimit(100000)
#
#import pickle
#from datetime import datetime
#
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'company_topics_ordered_df'+'_'+timestamp
#with open('data/'+filename+'.pickle', 'wb') as to_write:
#    pickle.dump(company_topics_ordered_df, to_write)
#%%
###############################################################################
####################################End########################################
###############################################################################
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
import sys
sys.setrecursionlimit(100000)

import pickle
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename = 'company_df'+'_'+timestamp
with open('data/'+filename+'.pickle', 'wb') as to_write:
    pickle.dump(company_df, to_write)
#%%
filename = 'data/' + 'company_df_2019-12-01-12-23-05.pickle'
with open(filename,'rb') as read_file:
    company_df = pickle.load(read_file)
#%%
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename = 'company_topics_ordered_df'+'_'+timestamp
company_topics_ordered_df.to_csv(r'data/'+filename+'.csv')
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
##%%
company_topics_df_filtered = company_topics_df[company_topics_df.topic_max >= 0.10]
company_topics_df_filtered.reset_index(drop=True,inplace=True)
company_topics_df_filtered.shape
#%%
sorted(Counter(company_topics_df_filtered.cluster).items())
#%%
################################Sort in Order##################################
#%%
def reverse(array): 
    array_reversed = array[::-1] 
    return array_reversed 
#%%
company_transcripts_ordered = list(reverse(company_df.transcript_name.unique()))
company_transcripts_ordered
#%%
transcript_name_index = []
for transcript_name in company_topics_sum_df.transcript_name:
    transcript_name_index.append(company_transcripts_ordered.index(transcript_name))
#%%
company_topics_ordered_df = company_topics_sum_df.copy()
company_topics_ordered_df['transcript_name_index'] = transcript_name_index
company_topics_ordered_df.sort_values(by=['transcript_name_index'],inplace=True)
company_topics_ordered_df.reset_index(drop=True,inplace=True)
company_topics_ordered_df
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%