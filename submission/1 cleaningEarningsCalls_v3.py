#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:56:44 2019

@author: lindaxju
"""

#%%
import pandas as pd

import time
seconds = 1
from bs4 import BeautifulSoup, NavigableString

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

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
###########################Clean Soup of Transcript############################
#%%
def get_soup_elements(s):
    
    """
    Input is soup of transcript page
    Returns list of all content in soup (regardless of tags)
    """
    
    soup_elements = []
    soup_body = s.find('body')
    
    for element in soup_body.next_elements:
        if isinstance(element, NavigableString) and len(str(element)) > 1:
            soup_elements.append(element.strip())
    
    return soup_elements
#%%
def get_tag_content(s,tag):
    
    """
    Input is soup of transcript page and tag of interest
    Returns list of content in soup for the tag of interest
    """
    
    tag_content = []
    
    separate_tag_contents = s.find_all(tag) # list of content for each tag
    
    for el in separate_tag_contents: # clean content and add to list
        tag_content.append(el.get_text().strip())
        
    return tag_content
#%%
def get_section_dict(s_e,sections):
        
    """
    Input is list of all content in soup (regardless of tags)
    Returns dictionary with each section as key and content of section as value
    """
    
    num_sections = len(sections)
    sections_dict = dict()
    
    for i in range(num_sections):
        section1 = sections[i]
        section1_index = s_e.index(section1)
        if i < num_sections-1: # need to account for last section
            section2 = sections[i+1]
            section2_index = s_e.index(section2)
        else:
            section2_index = len(s_e)
        sections_dict[section1] = s_e[section1_index+1:section2_index]
    
    return sections_dict
#%%
def print_dict_of_dict_keys(dict_of_dict):
    
    """
    Input is a dictionary of dictionaries
    Prints keys of the outer and inner dictionaries
    """
    
    for key1, val1 in dict_of_dict.items():
        print(key1)
        counter = 0
        for key2, val2 in val1.items():
            print(key2)
            counter += 1
        print(counter)
        print('')
#%%
def clean_soup_in_dict_of_dict(dict_of_dict):
    
    """
    Input is a dictionary of dictionaries
    Returns the dictionary of dictionaries with clean soups (paragraphs listed in sections)
    """
    
    dict_all_clean = dict(dict_of_dict.copy())
    
    for company, dict_transcript in dict_of_dict.items(): # loop through each company
        
        print(company)
        dict_transcript_clean = dict(dict_transcript.copy()) # convert from default dict to dict
        
        for transcript, list_soup_transcript_page in dict_transcript.items(): # loop through each transcript soup
            print(transcript)
            soup_elements =  get_soup_elements(list_soup_transcript_page[0]) # get all soup elements
            sections = get_tag_content(list_soup_transcript_page[0],'h2') # get sections
            section_dict = get_section_dict(soup_elements,sections) # get dictionary of contents by section
            dict_transcript_clean[transcript] = section_dict # reassign to transcript dictionary
        
        dict_all_clean[company] = dict_transcript_clean # reassign to company dictionary
        
        print('')
    
    return dict_all_clean
#%%
##################################Stop Words###################################
#%%
def get_stop_words(participant_names_orig,nicknames,stop_words_company_specific,stop_words_other):
        
    """
    Input is names of participants in original form, list of nicknames, specific stop words for the company, and other general stop words
    Returns a set of all stop words for a particular transcript
    """
    
    stop_words_english = stopwords.words('english') # regular stop words
    stop_words_english.remove('no') # remove no and not
    stop_words_english.remove('not') 
    
    participant_names_clean = [] # only get names of participants
    for i in range(len(participant_names_orig)):
        if i % 2 == 0:
            participant_names_clean.append(participant_names_orig[i].split()[0].lower())
    
    stop_words_nicknames = [] # get all names and nicknames
    for nickname in nicknames:
        if nickname[0] in participant_names_clean:
            stop_words_nicknames.append(nickname)
    stop_words_nicknames = [val for sublist in stop_words_nicknames for val in sublist] # flatten list
    stop_words_nicknames += participant_names_clean # add all of the original names
    
    stop_words_all = set(stop_words_english + stop_words_nicknames + stop_words_company_specific + stop_words_other)
    stop_words_all = sorted(stop_words_all)

    return stop_words_all
#%%
#####################################Clean#####################################
#%%
def get_transcript_corporate(transcript,participants_corporate,participants_noncorporate):
    
    """
    Input is a section of the transcript, corporate participant names, and non-corporate participant names
    Returns the corporate and non-corporate dialogue
    """
    
    transcript_corporate = []
    transcript_noncorporate = []
    corporate_dialogue = False # set initially as False
    
    for element in transcript: # loop through each transcript element
        if element in participants_corporate: # set to True
            corporate_dialogue = True
            continue
        elif element in participants_noncorporate: # set to False
            corporate_dialogue = False
            continue
        if corporate_dialogue:
            transcript_corporate.append(element) # add to corporate list
        else:
            transcript_noncorporate.append(element) # add to not corporate list
    
    return transcript_corporate,transcript_noncorporate
#%%
def get_cleaned_transcript(transcript,stop_phrases,lemmatizer_exceptions,stop_word_list):
        
    """
    Input is a transcript and list of stop words
    Returns lists of the original transcript and the cleaned transcript
    """
    
    transcript_orig = []
    transcript_cleaned = []
    
    for element in transcript: # loop through each transcript element
    
        element_clean1 = element.replace("n't"," not")
        element_clean1 = re.sub(r'[^a-zA-Z]+',' ', element_clean1) # remove punctuation
        element_clean1 = element_clean1.lower()
        
        for stop_phrase in stop_phrases:
            element_clean1 = element_clean1.replace(stop_phrase+' ', '')
        
        lemmatizer = WordNetLemmatizer()
        
        element_clean2 = ''
        
        for word in element_clean1.split(): # loop through every word
            if word not in lemmatizer_exceptions: # words like goods should not be lemmatized to good
                new_word = lemmatizer.lemmatize(word)
            else:
                new_word = word
            if new_word not in stop_word_list and len(new_word) >= 2: # len >= ensures no 1 letter words ('re and 've are filtered out for 3)
                element_clean2 += new_word + ' '
        
        element_clean2 = element_clean2.strip()
        
        if len(element_clean2) >= 50: # len >= ensures at least a sentence
            transcript_orig.append(element)
            transcript_cleaned.append(element_clean2)
            
    return transcript_orig, transcript_cleaned
#%%
def get_df_company_presentation(company,stop_phrases,stop_words_company_specific,stop_words_other,lemmatizer_exceptions):
    
    """
    Input is dictionary of company transcripts, company-specific stop words, and general stop words
    Returns dataframe of transcript name, original paragraph, and cleaned paragraph (presentation only)
    """
    
    transcript_name_presentation = []
#    transcript_name_qa = []
    transcript_presentation_corporate_orig_company = []
#    transcript_qa_corporate_orig_company = []
    transcript_presentation_corporate_cleaned_company = []
#    transcript_qa_corporate_cleaned_company = []
    
    for transcript_key in company.keys():
        dict_transcript = company[transcript_key].copy()
        try:
            participants_corporate = dict_transcript['Corporate Participants'].copy()
            participants_noncorporate = dict_transcript['Conference Call Participants'].copy()
            transcript_presentation = dict_transcript['Presentation'].copy()
#           transcript_qa = dict_transcript['Questions and Answers'].copy()
        except:
            continue
        
        participants = participants_corporate + participants_noncorporate
        stop_words_all = get_stop_words(participants,nicknames,stop_words_company_specific,stop_words_other)
        
        # get transcript split by corporate and noncorporate dialogue for presentation and Q&A sections
        transcript_presentation_corporate,transcript_presentation_noncorporate = get_transcript_corporate(transcript_presentation,participants_corporate,participants_noncorporate)
#        transcript_qa_corporate,transcript_qa_noncorporate = get_transcript_corporate(transcript_qa,participants_corporate,participants_noncorporate)
    
        # get original and cleaned transcript for presentation and Q&A sections
        transcript_presentation_corporate_orig,transcript_presentation_corporate_cleaned = get_cleaned_transcript(transcript_presentation_corporate,stop_phrases,lemmatizer_exceptions,stop_words_all)
#        transcript_qa_corporate_orig,transcript_qa_corporate_cleaned = get_cleaned_transcript(transcript_qa_corporate,stop_phrases,lemmatizer_exceptions,stop_words_all)
        
        # add cleaned presentation paragraphs to running list
        transcript_presentation_corporate_orig_company += transcript_presentation_corporate_orig
        transcript_presentation_corporate_cleaned_company += transcript_presentation_corporate_cleaned
        
        # add cleaned Q&A paragraphs to running list
#        transcript_qa_corporate_orig_company += transcript_qa_corporate_orig
#        transcript_qa_corporate_cleaned_company += transcript_qa_corporate_cleaned
        
        # keep tally of transcript
        transcript_name_presentation += [transcript_key] * len(transcript_presentation_corporate_orig)
#        transcript_name_qa += [transcript_key] * len(transcript_qa_corporate_orig)
    
    df_company_presentation = pd.DataFrame(list(zip(transcript_name_presentation,transcript_presentation_corporate_orig_company,transcript_presentation_corporate_cleaned_company)),columns =['transcript_name','content_orig','content_cleaned']) 
#    df_company_qa = pd.DataFrame(list(zip(transcript_name_qa,transcript_qa_corporate_orig_company,transcript_qa_corporate_cleaned_company)),columns =['transcript_name','content_orig','content_cleaned']) 

    return df_company_presentation#,df_company_qa
#%%
def check_stop_paragraphs(stop_paragraph,content_cleaned):
    
    """
    Input is words in stop paragraphs
    Returns True if content_cleaned has all words and should be removed
    """
    
    for word in stop_paragraph:
        if word in content_cleaned:
            remove_one = True
        else:
            remove_one = False
            break
    
    return remove_one
#%%
###############################################################################
################################Code Execution#################################
###############################################################################
#%%
#############################Import Scraped Soups##############################
#%%
import sys
sys.setrecursionlimit(100000)

import pickle

filename = 'data/' + 'dict_department_stores_2019-12-07-15-41-17.pickle'
with open(filename,'rb') as read_file:
    dict_all = pickle.load(read_file)
#%%
dict_all_clean = clean_soup_in_dict_of_dict(dict_all)
print_dict_of_dict_keys(dict_all_clean)
#%%
dict_all_clean.keys()
#%%
nicknames = [] # import nicknames
with open('data/' + 'names.txt') as inputfile:
    for line in inputfile:
        nicknames.append(line.strip().split(','))

stop_phrases = ['good morning','good afternoon','good evening']
stop_words_other = ['thank','billion','million','dollar','morning','afternoon',
                    'question','answer','quarter','approximately','reminder',
                    'presentation','discussion','welcome','was']
lemmatizer_exceptions = ['goods']
#%%
# JCP
stop_words_JCP = []
# KSS
stop_words_KSS = []
# M
stop_words_M = ['hal','karen','hoguet'] # assumes Macy's
# JWN
stop_words_JWN = []
#%%
company_all_df = pd.DataFrame(columns=['transcript_name','content_orig','content_cleaned'])
for company_key in list(dict_all_clean.keys()):
    company = dict_all_clean[company_key].copy()
    company_all_df_new = get_df_company_presentation(company,stop_phrases,stop_words_M,stop_words_other,lemmatizer_exceptions)
    company_all_df = pd.concat([company_all_df,company_all_df_new])
company_all_df.reset_index(drop=True,inplace=True)
company_all_df
#%%
# get rid of uninformative paragraphs
company_df_transcript_name = []
company_df_content_orig = []
company_df_content_cleaned = []
stop_paragraphs = [['transcription','reproduction','statement','made','call','without','consent','replay','available','website','concludes'],
                   ['president','chief','financial','officer','discus','performance'],
                   ['forward','looking','statement','future','anticipate','current'],
                   ['please','note','portion','rebroadcast','without','prior','written','updated','possible','information','longer','current'],
                   ['joining','today','earnings','call','last','minute'],
                   ['please','refer','investor','relation','section','reconciliation','non','gaap','financial','measure','discussed'],
                   ['turn','call','result','add','update','key','initiative'],
                   ['first','want','ask','person','ask','one','additional','queue','follow','order','give'],
                   ['slide','supplemental','website','meant','facilitate','company','result','used','reference','document','following','call'],
                   ['slide','investor','relation','want','mention','viewed','going','section'],
                   ['forward','looking','statement','subject','uncertainty','materially','assumption','factor','variety','risk','company','recent','form'],
                   ['joining','today','call','ceo','cfo','following','prepared','remark','forward','taking'],
                   ['forward','looking','statement','within','meaning','private','security','litigation','reform','act'],
                   ['call','scheduled','discus','cfo','company'],
                   ['call','chief','executive','officer','president']
                   ]

for row in range(len(company_all_df)):
    
    transcript_name_temp = company_all_df.transcript_name[row]
    content_orig_temp = company_all_df.content_orig[row]
    content_cleaned_temp = company_all_df.content_cleaned[row]
    
    remove = False
    
    for stop_paragraph in stop_paragraphs:
        remove = check_stop_paragraphs(stop_paragraph,content_cleaned_temp)
        if remove:
            break
    
    if not remove:
        company_df_transcript_name.append(transcript_name_temp)
        company_df_content_orig.append(content_orig_temp)
        company_df_content_cleaned.append(content_cleaned_temp)

company_df = pd.DataFrame(list(zip(company_df_transcript_name,company_df_content_orig,company_df_content_cleaned)),columns =['transcript_name','content_orig','content_cleaned'])
company_df
#%%
#import sys
#sys.setrecursionlimit(100000)
#
#import pickle
#from datetime import datetime
#
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'company_df'+'_'+timestamp
#with open('data/'+filename+'.pickle', 'wb') as to_write:
#    pickle.dump(company_df, to_write)
#%%
#from datetime import datetime
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'company_df'+'_'+timestamp
#company_df.to_csv(r'data/'+filename+'.csv')
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
#%%
#%%
#%%
#%%
#%%
#%%
#%%
num_components = 20
vectorizer_tfidf_presentation,nmf_tfidf_presentation,company_df = get_topics(company_df,num_components)
display_topics(nmf_tfidf_presentation,vectorizer_tfidf_presentation.get_feature_names(),5)
#%%
company_df.shape
#%%
company_df_filtered = company_df[company_df.topic_max >= 0.10]
company_df_filtered.reset_index(drop=True,inplace=True)
company_df_filtered.shape
#%%
sorted(Counter(company_df_filtered.cluster).items())
#%%
#%%
#%%
#%%
filename = 'data/' + 'company_df_2019-12-01-12-23-05.pickle'
with open(filename,'rb') as read_file:
    company_df = pickle.load(read_file)
#%%
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename = 'company_df'+'_'+timestamp
company_df.to_csv(r'data/'+filename+'.csv')
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
soup_tags_content = ['h1','h2','p']
soup_tags_participants = ['h3','strong','em']
#%%
from selenium import webdriver
import chromedriver_binary

browser = webdriver.Chrome()

logged_in_test = False
url_test = 'https://www.bamsec.com/transcripts/12879683'

soup_test = get_soup_transcript_page(url_test,logged_in_test)
#%%
soup_elements =  get_soup_elements(soup_test)
sections = get_tag_content(soup_test,'h2')
Q3_2019_Macys_Inc_Earnings_Call = get_section_dict(soup_elements,sections)
#%%
Q3_2019_Macys_Inc_Earnings_Call
#%%
Q3_2019_Macys_Inc_Earnings_Call.keys()
#%%
Q3_2019_Macys_Inc_Earnings_Call['Questions and Answers'][2]
#%%
browser.close()
#%%
dict_all_clean = dict(dict_all.copy()) # convert from default dict to dict

for company, dict_transcript in dict_all.items(): # loop through each company
    print(company)
    dict_transcript_clean = dict(dict_transcript.copy()) # convert from default dict to dict
    
    for transcript, list_soup_transcript_page in dict_transcript.items(): # loop through each transcript soup
        print(transcript)
        soup_elements =  get_soup_elements(list_soup_transcript_page[0]) # get all soup elements
        sections = get_tag_content(list_soup_transcript_page[0],'h2') # get sections
        section_dict = get_section_dict(soup_elements,sections) # get dictionary of contents by section
        dict_transcript_clean[transcript] = section_dict # reassign to transcript dictionary
    
    dict_all_clean[company] = dict_transcript_clean # reassign to company dictionary
    
    print('')
#%%
print_dict_of_dict_keys(dict_all_clean)
#%%
##################################Stop Words###################################
#%%
# regular stop words
stop_words_english = stopwords.words('english')
stop_words_english
#%%
# first names and nicknames of participants
stop_words_names = []

for i in range(len(test_participants_corporate)):
    if i % 2 == 0:
        stop_words_names.append(test_participants_corporate[i].split()[0].lower())
        
for i in range(len(test_participants_noncorporate)):
    if i % 2 == 0:
        stop_words_names.append(test_participants_noncorporate[i].split()[0].lower())
        
stop_words_names
#%%
nicknames = []
with open('data/' + 'names.txt') as inputfile:
    for line in inputfile:
        nicknames.append(line.strip().split(','))
        
nicknames[:10]
#%%
stop_words_nicknames = []

for nickname in nicknames:
    if nickname[0] in stop_words_names:
        stop_words_nicknames.append(nickname)
        
stop_words_nicknames = [val for sublist in stop_words_nicknames for val in sublist]

stop_words_nicknames
#%%
stop_words_M = ['hal']
stop_words_other = ['thank','billion','million','dollar','morning','afternoon',
                    'question','answer','quarter','approximately','reminder',
                    'presentation','discussion']
#%%
stop_words_all = set(stop_words_english + stop_words_nicknames + stop_words_M + stop_words_other)
stop_words_all
#%%
seconds = 1

test_transcript_corporate = []
corporate_dialogue = False # set initially as False

for element in test_transcript: # loop through each transcript element
    
    if element in test_participants_corporate: # set to True
        corporate_dialogue = True
        continue
    elif element in test_participants_noncorporate: # set to False
        corporate_dialogue = False
        continue
    if corporate_dialogue:
        test_transcript_corporate.append(element) # add to list

test_transcript_corporate
#%%
test_transcript_corporate_orig = []
test_transcript_corporate_cleaned = []

for element in test_transcript_corporate: # loop through each transcript element
    
    element_clean1 = element.replace("n't"," not")
    element_clean1 = re.sub(r'[^a-zA-Z]+',' ', element_clean1) # remove punctuation
    element_clean1 = element_clean1.lower()
    
    stop_words_english = set(stopwords.words('english') + stop_words_nicknames)
    lemmatizer = WordNetLemmatizer()
    
    element_clean2 = ''
    
    for word in element_clean1.split(): # loop through every word
            new_word = lemmatizer.lemmatize(word)
        if new_word not in stop_words_all and len(new_word) >= 3: # len >= ensures 're and 've are filtered out
            element_clean2 += new_word + ' '
    
    element_clean2 = element_clean2.strip()
    
    if len(element_clean2) >= 50: # len >= ensures at least a sentence
        test_transcript_corporate_orig.append(element)
        test_transcript_corporate_cleaned.append(element_clean2)

test_transcript_corporate_orig
test_transcript_corporate_cleaned
#%%
# CountVectorizer
vectorizer_cv, doc_word_cv = get_doc_word_vectorizer(vectorizer='cv',ngram_range=(1,2),list_docs=test_transcript_corporate_cleaned)
get_doc_word_df(doc_word=doc_word_cv,vectorizer=vectorizer_cv,list_docs=test_transcript_corporate_cleaned)
#%%
# Acronynms: Latent Semantic Analysis (LSA) is just another name for 
#  Signular Value Decomposition (SVD) applied to Natural Language Processing (NLP)
lsa_tfidf, doc_topic_lsa_tfidf = get_dim_red(svd='lsa',doc_word=doc_word_tfidf,num_components=num_components)
lsa_tfidf.explained_variance_ratio_
#%%
display_topics(lsa_tfidf,vectorizer_tfidf.get_feature_names(),5)
#%%
Vt = get_doc_topic_df(doc_topic=doc_topic_lsa_tfidf,transcript_orig=transcript_corporate_orig,transcript_cleaned=transcript_corporate_cleaned,cols=list_components,rnd=3)
Vt
#%%
sorted(Counter(Vt.cluster).items())
#%%
def get_doc_topic_df(doc_topic,transcript_orig,transcript_cleaned,cols,rnd=3):

    """
    Input is doc topic object, original transcript, cleaned transcript, and names of columns
    Returns dataframe of original, clean, and topic of transcript
    """
    
    doc_topic_df = pd.DataFrame(doc_topic.round(rnd),columns=cols)
    doc_topic_df['topic_max'] = doc_topic_df.max(axis=1)
    doc_topic_df['cluster'] = doc_topic_df.idxmax(axis=1)
    docs_df_orig = pd.DataFrame(transcript_orig,columns=['content_orig'])
    docs_df_clean = pd.DataFrame(transcript_cleaned,columns=['content_clean'])
    doc_topic_df = pd.concat([docs_df_orig,docs_df_clean,doc_topic_df],axis=1)
    
    return doc_topic_df
#%%
test_company_key = ('Macys', 'https://www.bamsec.com/companies/794367/macy-s-inc/transcripts')
test_company = dict_all_clean[test_company_key].copy()
test_company.keys()
#%%
test_dict_transcript_key = ("Q3 2019 Macy's Inc Earnings Call", '11/21/19', 'https://www.bamsec.com/transcripts/12879683')
test_dict_transcript = test_company[test_dict_transcript_key].copy()
test_dict_transcript.keys()
#%%
test_dict_transcript_key2 = 'Presentation'
test_transcript = test_dict_transcript[test_dict_transcript_key2].copy()
test_transcript
#%%
test_dict_transcript_key2 = 'Questions and Answers'
test_qa = test_dict_transcript[test_dict_transcript_key2].copy()
test_qa
#%%
test_dict_transcript_key2 = 'Corporate Participants'
test_participants_corporate = test_dict_transcript[test_dict_transcript_key2].copy()
test_participants_corporate
#%%
test_dict_transcript_key2 = 'Conference Call Participants'
test_participants_noncorporate = test_dict_transcript[test_dict_transcript_key2].copy()
test_participants_noncorporate.append('Operator')
test_participants_noncorporate
#%%
participant_names_orig = test_participants_corporate + test_participants_noncorporate
stop_words_M = ['hal']
stop_words_other = ['thank','billion','million','dollar','morning','afternoon',
                    'question','answer','quarter','approximately','reminder',
                    'presentation','discussion']
stop_words_all = get_stop_words(participant_names_orig,nicknames,stop_words_M,stop_words_other)
#%%
transcript = test_transcript 
#transcript = test_qa
participants_corporate = test_participants_corporate
participants_noncorporate = test_participants_noncorporate
transcript_corporate,transcript_noncorporate = get_transcript_corporate(transcript,participants_corporate,participants_noncorporate)
#%%
transcript_corporate_orig,transcript_corporate_cleaned = get_cleaned_transcript(transcript_corporate,stop_words_all)
#%%
print(len(transcript_corporate_orig))
print(len(transcript_corporate_cleaned))
#%%
# TF-IDF
vectorizer_tfidf, doc_word_tfidf = get_doc_word_vectorizer(vectorizer='tfidf',ngram_range=(1,2),list_docs=transcript_corporate_cleaned)
get_doc_word_df(doc_word=doc_word_tfidf,vectorizer=vectorizer_tfidf,list_docs=transcript_corporate_cleaned)
#%%
num_components = 10
list_components = get_list_components(num_components)
#%%
# NMF
nmf_tfidf, doc_topic_nmf_tfidf = get_dim_red(svd='nmf',doc_word=doc_word_tfidf,num_components=num_components)
#%%
display_topics(nmf_tfidf,vectorizer_tfidf.get_feature_names(),5)
#%%
H = get_doc_topic_df(doc_topic=doc_topic_nmf_tfidf,transcript_orig=transcript_corporate_orig,transcript_cleaned=transcript_corporate_cleaned,cols=list_components,rnd=3)
H
#%%
H_filtered = H[H.topic_max >= 0.10]
H_filtered.reset_index(drop=True,inplace=True)
#%%
sorted(Counter(H.cluster).items())
#%%
H_filtered = H[H.topic_max >= 0.10]
H_filtered.reset_index(drop=True,inplace=True)
H_filtered
#%%
sorted(Counter(H_filtered.cluster).items())
#%%
#company_key = ('JCP', 'https://www.bamsec.com/companies/1166126/j-c-penney-co-inc/transcripts')
#company_key = ('KSS', 'https://www.bamsec.com/companies/885639/kohl-s-corp/transcripts')
company_key = ('M', 'https://www.bamsec.com/companies/794367/macy-s-inc/transcripts') # should produce company_df with 2508 rows and 3 columns
#company_key = ('JWN', 'https://www.bamsec.com/companies/72333/nordstrom-inc/transcripts')
company = dict_all_clean[company_key].copy()
company.keys()
#%%
company_df = get_df_company_presentation(company,stop_phrases,stop_words_M,stop_words_other,lemmatizer_exceptions)
company_df
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