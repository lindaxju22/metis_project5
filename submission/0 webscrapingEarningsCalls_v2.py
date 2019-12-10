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
from selenium import webdriver
import chromedriver_binary
from bs4 import BeautifulSoup, NavigableString

from collections import defaultdict
#import re
#%%
###############################################################################
##############################Helper Functions#################################
###############################################################################
#%%
#####################Scrape Company Page and Get Links#########################
#%%
def get_soup_company_page(url,logged_in):
    
    """
    Input is url of company transcripts page and whether browser is logged in
    Returns soup of company transcripts page
    """
    
    browser.get(url)
    
    if logged_in == False:
        browser.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[4]/a').click()
        user = 'lindaxju@gmail.com'
        pw = 'jCz#066$'
        browser.find_element_by_id('email').send_keys(user)
        browser.find_element_by_id('password').send_keys(pw)
        browser.find_element_by_xpath('/html/body/div[2]/div/div/div[1]/div/form/button').click()
    
    source_data = browser.page_source
    soup = BeautifulSoup(source_data,'html.parser')
    
    return soup
#%%
def get_earnings_links(s):
    
    """
    Input is soup of company transcripts page
    Returns dataframe with three columns: title of call, date of call, and 
     transcript link
    """
    
    list_titles = []
    list_dates = []
    list_links = []

    label_left = 'Earnings' # looking for Earnings transcripts
    soup_body = s.find('body') # look in body only
    separate_tag_contents = soup_body.find_all('a','list-group-item single-line transcript') # earnings links are in here
    
    for element in separate_tag_contents: # loop through content
        transcript_type = element.find('span','label-left').get_text().strip()
        if transcript_type == label_left: # check if Earnings transcript
            # find and add title
            tag1 = element.find('span','label-center with-right-label with-left-label')
            transcript_title = tag1.find('span').nextSibling.strip()
            list_titles.append(transcript_title) 
            # find and add date
            date = element.find('span','label-right').get_text().strip()
            list_dates.append(date)
            # find and add link
            list_links.append(element['href']) 
            dict_temp = {'title':list_titles,'date':list_dates,'link':list_links} # create dict
            df_earnings_links = pd.DataFrame(dict_temp) # create df from dit
            
    return df_earnings_links
#%%
##########################Scrape Single Earnings Call##########################
#%%
def get_soup_transcript_page(url,logged_in):
    
    """
    Input is url of transcript page and whether browser is logged in
    Returns soup of transcript page
    """
    
    browser.get(url)
    
    if logged_in == False:
        user = 'lindaxju@gmail.com'
        pw = 'jCz#066$'
        browser.find_element_by_id('email').send_keys(user)
        browser.find_element_by_id('password').send_keys(pw)
        browser.find_element_by_xpath('/html/body/div[3]/div/div/div[1]/div/form/button').click()
    
    source_data = browser.page_source
    soup = BeautifulSoup(source_data,'html.parser')
    
    iframe_src = soup.select_one("#embedded_doc").attrs["src"]
    browser.get(iframe_src)
    
    source_data = browser.page_source
    soup = BeautifulSoup(source_data, "html.parser")
    
    return soup
#%%
###############################################################################
################################Code Execution#################################
###############################################################################
#%%
dict_all = defaultdict()

companies = [('JCP', 'https://www.bamsec.com/companies/1166126/j-c-penney-co-inc/transcripts'),
             ('KSS','https://www.bamsec.com/companies/885639/kohl-s-corp/transcripts'),
             ('M','https://www.bamsec.com/companies/794367/macy-s-inc/transcripts'),
             ('JWN','https://www.bamsec.com/companies/72333/nordstrom-inc/transcripts')]
logged_in = False
counter_company = 0

browser = webdriver.Chrome()

for company in companies: # loop through all companies
    
    ticker_company = company[0]
    url_company = company[1]
    dict_transcript = defaultdict()
    print(url_company)
    
    soup_company_page = get_soup_company_page(url_company,logged_in) # get soup of company page
    logged_in = True
    
    df_links_company = get_earnings_links(soup_company_page) # get transcript links from company page
    
    counter_transcript = 0
    
    for row in df_links_company.itertuples(): # loop through all transcript links
        
        title = row.title
        date = row.date
        link = row.link
        url_base = 'https://www.bamsec.com'
        url_transcript = url_base+link
        transcript = (title,date,url_transcript,ticker_company) # save key
        
        print(url_transcript)
        soup_transcript_page = get_soup_transcript_page(url_transcript,logged_in)
        dict_transcript[transcript] = [soup_transcript_page] # save soup
        
        counter_transcript += 1
        print("{} of {} transcripts scraped".format(counter_transcript,df_links_company.shape[0]))
        
        time.sleep(seconds)
    
    dict_all[company] = dict(dict_transcript) # save all transcripts for a company and convert back to dict
    
    counter_company += 1
    print("{} of {} company pages scraped".format(counter_company,len(companies)))
    
    time.sleep(seconds)

dict_all = dict(dict_all.copy())
#%%
#dict_all
#%%
for key, value in dict_all.items():
    print(key)
    print(type(value))
    new_dict = value
    for key, value in new_dict.items():
        print(key)
        print(type(value))
#%%
browser.close()
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#import sys
#sys.setrecursionlimit(100000)
#
#import pickle
#from datetime import datetime
#
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'dict_department_stores'+'_'+timestamp
#with open('data/'+filename+'.pickle', 'wb') as to_write:
#    pickle.dump(dict_all, to_write)
##%%
#filename = 'data/' + 'dict_all_2019-11-24-15-21-10.pickle'
#with open(filename,'rb') as read_file:
#    check_dict_all = pickle.load(read_file)
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
browser = webdriver.Chrome()
url = 'https://www.bamsec.com/companies/794367/macy-s-inc/transcripts'
browser.get(url)

browser.find_element_by_xpath('/html/body/div[1]/div/div[2]/ul/li[4]/a').click()
user = 'lindaxju@gmail.com'
pw = 'jCz#066$'
browser.find_element_by_id('email').send_keys(user)
browser.find_element_by_id('password').send_keys(pw)
browser.find_element_by_xpath('/html/body/div[2]/div/div/div[1]/div/form/button').click()

source_data = browser.page_source
soup = BeautifulSoup(source_data,'html.parser')
#%%
soup
#%%
browser.close()
#%%
browser = webdriver.Chrome()
url = 'https://www.bamsec.com/transcripts/12879683'
browser.get(url)

user = 'lindaxju@gmail.com'
pw = 'jCz#066$'
browser.find_element_by_id('email').send_keys(user)
browser.find_element_by_id('password').send_keys(pw)
browser.find_element_by_xpath('/html/body/div[3]/div/div/div[1]/div/form/button').click()

source_data = browser.page_source
soup = BeautifulSoup(source_data,'html.parser')

iframe_src = soup.select_one("#embedded_doc").attrs["src"]
browser.get(iframe_src)

source_data = browser.page_source
soup = BeautifulSoup(source_data, "html.parser")
#%%
soup
#%%
browser.close()
#%%
# generate list to fill
num_tags_content = len(soup_tags_content)
list_transcript = [[] for i in range(num_tags_content)]

# loop and fill list with content of tags
for i in range(num_tags_content):
    tag = soup_tags_content[i]
    list_tags_content = list(soup.find_all(tag)) # list of content for each tag
    for el in list_tags_content: # clean content and add to list
        el_cleaned = el.get_text().strip()
        if len(el_cleaned) > 0:
            list_transcript[i].append(el_cleaned)
            
list_transcript
#%%
# alternative but fills list with 'body', not tag type:
# generate list to fill
soup_elements = []
soup_body = soup.find('body')

for element in soup_body.next_elements:
    if isinstance(element, NavigableString) and len(str(element)) > 1:
        soup_elements.append(element.strip())

soup_elements
#%%
list_titles = []
list_dates = []
list_links = []

soup_body = soup.find('body')
separate_tag_contents = soup_body.find_all('a','list-group-item single-line transcript')
label_left = 'Earnings' # looking for Earnings transcripts

for element in separate_tag_contents: # loop through content and check if Earnings transcript
    print('---')
    transcript_type = element.find('span','label-left').get_text().strip()
    print(transcript_type)
    if transcript_type == label_left:
        tag1 = element.find('span','label-center with-right-label with-left-label')
        transcript_title = tag1.find('span').nextSibling.strip()
        date = element.find('span','label-right').get_text().strip()
        list_titles.append(transcript_title) # add to list
        list_dates.append(date) # add to list
        list_links.append(element['href']) # add to list
        print(transcript_title)
        print(date)
        dict = {'title':list_titles,'date':list_dates,'link':list_links} # create dataframe
        df_earnings_links = pd.DataFrame(dict) 
#%%
df_earnings_links
#%%
df_links_macys = get_earnings_links(soup)
df_links_macys
#%%
soups_macys = []
logged_in = False
counter = 0

browser = webdriver.Chrome()

for link in df_links_macys.link:
    url_base = 'https://www.bamsec.com'
    url = url_base+link
    
    soup = get_soup_transcript_page(url,logged_in)
    soups_macys.append(soup)
    
    counter += 1
    logged_in = True
    print("{} of {} transcripts scraped".format(counter,len(df_links_macys.link)))
    print(url)
    
    time.sleep(seconds)
#%%
browser.close()
#%%
soups_macys
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