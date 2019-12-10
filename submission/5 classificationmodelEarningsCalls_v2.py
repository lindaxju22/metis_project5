#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:56:44 2019

@author: lindaxju
"""

#%%
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from datetime import datetime, timedelta
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
#%%
###############################################################################
#################################Data Setup####################################
###############################################################################
#%%
########################Import Features and Target#############################
#%%
filename = 'data/' + 'classification_df_2019-12-07-19-31-02.pickle'
with open(filename,'rb') as read_file:
    classification_df = pickle.load(read_file)
#%%
classification_df.sort_values(by=['date'],inplace=True)
classification_df.reset_index(drop=True,inplace=True)
classification_df.head()
#%%
Counter(classification_df.buy_signal) #slight imbalance
#%%
classification_df.columns
#%%
###############################Choosing Features###############################
#%%
X_cols_list = ['per_negative','per_positive','per_uncertainty','per_litigious',
               'per_constraining','per_interesting','per_modal1','per_modal2',
               'per_modal3','c01','c02', 'c03','c04','c05','c06','c07','c08',
               'c09','c10','c11','c12','c13','c14','c15','c16','c17','c18',
               'c19','c20']
#%%
X_orig = classification_df[X_cols_list]
X_orig.info()
X_orig.describe()
#%%
# visualize distributions
X_orig.hist(bins=30,figsize=(20,20));
#%%
###############################Fixing Imbalance################################
#%%
y_orig = classification_df['buy_signal']
y_orig.describe()
print(Counter(y_orig))
pd.Series(y_orig).value_counts().plot('bar',figsize=(4,3),title='Buy Signal (Full Dataset)',rot=360)
#%%
test_size_perc = 0.20
test_size_num = round(len(X_orig) * test_size_perc)
train_val_size_num = len(X_orig) - test_size_num
val_size_perc = 0.20
val_size_num = round(train_val_size_num * test_size_perc)
train_size_num = train_val_size_num - val_size_num

X_train_val_orig = X_orig.copy().iloc[:train_val_size_num]
y_train_val_orig = y_orig.copy()[:train_val_size_num]
y_train_val_orig = y_train_val_orig.astype('int')
X_train_orig = X_train_val_orig.copy().iloc[:train_size_num]
y_train_orig = y_train_val_orig.copy()[:train_size_num]
y_train_orig = y_train_orig.astype('int')
X_val = X_train_val_orig.copy().iloc[train_size_num:]
y_val = y_train_val_orig.copy()[train_size_num:]
X_test = X_orig.copy().iloc[train_val_size_num:]
y_test = y_orig.copy()[train_val_size_num:]
#%%
# check imbalance
print("Train-Val {}".format(Counter(y_train_val_orig)))
print("Train {}".format(Counter(y_train_orig)))
print("Val {}".format(Counter(y_val)))
print("Test {}".format(Counter(y_test)))
#%%
# Now add some random oversampling of the minority classes
# Object to over-sample the minority class(es) by picking samples at random
# with replacement.
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_sample(X_train_orig,y_train_orig)
X_train_val, y_train_val = ros.fit_sample(X_train_val_orig,y_train_val_orig)
print(Counter(y_train))
pd.Series(y_train).value_counts().plot(kind='bar')
#%%
print("Train {}".format(Counter(y_train)))
print("Val {}".format(Counter(y_val)))
print("Test {}".format(Counter(y_test)))
#%%
pd.Series(y_train).value_counts().plot(kind='bar',figsize=(5,4),title='Buy Signal (Full Dataset)',rot=360)
#%%
pd.Series(y_val).value_counts().plot(kind='bar',figsize=(5,4),title='Buy Signal (Full Dataset)',rot=360)
#%%
pd.Series(y_test).value_counts().plot(kind='bar',figsize=(5,4),title='Buy Signal (Full Dataset)',rot=360)
#%%
####################Standardize and Set Default Threshold######################
#%%
# Standard scale - check with Roberto if need to do this; won't need for Random Forest
# Scale the predictors on train, validation, and test sets
# X_tr, y_train
# X_v, y_val
# X_te, y_test
#std = StandardScaler()
#std.fit(X_train)
#X_tr = std.transform(X_train)
#X_v = std.transform(X_val)
#X_te = std.transform(X_test)

X_tr = X_train.copy()
X_v = X_val.copy()
X_te = X_test.copy()
#%%
###############################################################################
##############################Helper Functions#################################
###############################################################################
#%%
############################Get y_pred from Model##############################
#%%
def get_y_pred(X_v,model,thresh):
            
    """
    Input is validation features, model, and proba_threshold
    Returns y_pred
    """
    
    predict_probas = model.predict_proba(X_v)
    max_index = predict_probas.argmax(axis=1)
    y_pred_list = []
    for i_pred in range(len(max_index)):
        i_class = max_index[i_pred]
        if predict_probas[i_pred][i_class] >= thresh:
            y_pred_list.append(model.classes_[i_class])
        else:
            y_pred_list.append(0)
    
    y_pred = np.asarray(y_pred_list)
    print(list(y_pred))
    
    return y_pred
#%%
###############################Return Functions################################
#%%
# define return metric
def get_return_dict(y_val,y_pred,df_returns_val,invest_amount=1000):
                
    """
    Input is actual y values, predicted y values, actual returns, and investment amount
    Returns dictionary of investment amount, dollar return, and percent return
    """
    
    return_dict = dict(invest_amt=0,return_dollar=0,return_perc=0)
    
    return_df = df_returns_val.copy()
    return_df['y_actual'] = y_val
    return_df['y_pred'] = y_pred
    return_df['invest_amt'] = -1 * y_pred * invest_amount
    return_df['return_dollar'] = -1 * return_df['invest_amt'] * (1 + return_df['returns']/100)
    
    return_dict['invest_amt'] = round(sum(return_df['invest_amt']),2)
    return_dict['return_dollar'] = round(sum(return_df['return_dollar']),2)
    if return_dict['invest_amt'] == 0 and return_dict['return_dollar'] != 0:
        return_dict['return_perc'] = 'arbitrage'
    elif return_dict['invest_amt'] == 0:
        return_dict['return_perc'] = np.nan
    else:
        return_dict['return_perc'] = round((return_dict['return_dollar']+return_dict['invest_amt'])/abs(return_dict['invest_amt'])*100,2)
    
    return return_dict
#%%
# extract validation returns
df_returns_val = pd.DataFrame(classification_df.loc[list(y_val.index)].returns)
df_returns_val
# extract test borrow rates
df_returns_test = pd.DataFrame(classification_df.loc[list(y_test.index)].returns)
df_returns_test
#%%
###############################################################################
###########################Classification Models###############################
###############################################################################
#%%
# set default threshold
threshold_default = 0.4
#%%
###############################Random Forest###################################
#%%
# tune parameter [n_estimators]
def tune_randomforest_n_estimators(range_n_estimators):
                    
    """
    Input is range_n_estimators
    Returns dictionary of return dictionaries
    """
    
    return_dict = defaultdict(int)
    
    for n in range_n_estimators:
        randomforest = RandomForestClassifier(n_estimators=n,oob_score=True,random_state=42)
        randomforest.fit(X_tr, y_train)

        y_pred = get_y_pred(X_val,randomforest,threshold_default)
        
        return_dict['n: '+str(n)] = get_return_dict(y_val,y_pred,df_returns_val)
        print(n)
        
    return return_dict
#%%
tune_randomforest_n_estimators(range(50,500,50))
# result: 400
#n_estimators_rf = 400
#%%
# testing
#y_pred_naive = np.array([1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1])
y_pred_naive = np.array([1]*len(y_val))
get_return_dict(y_val,y_pred_naive,df_returns_val,invest_amount=1000)
#%%
randomforest = RandomForestClassifier(n_estimators=400,oob_score=True,random_state=42)
randomforest.fit(X_train_val, y_train_val)
y_pred = get_y_pred(X_te,randomforest,threshold_default)
get_return_dict(y_test,y_pred,df_returns_test,invest_amount=1000)
#%%
# all buy
y_pred_naive = np.array([1]*len(y_test))
get_return_dict(y_test,y_pred_naive,df_returns_test,invest_amount=1000)
#%%
# all sell
y_pred_naive = np.array([-1]*len(y_test))
get_return_dict(y_test,y_pred_naive,df_returns_test,invest_amount=1000)
#%%
for i in range(len(list(y_test))):
    if list(y_pred)[i] != 0:
        print("pred {}, actual {}, return {}".format(list(y_pred)[i],list(y_test)[i],list(df_returns_test.returns)[i]))
#%%
###############################################################################
####################################End########################################
###############################################################################
#%%
#%%
#%%
###############################################################################
#%%
# tune parameter [max_depth]
def tune_randomforest_max_depth(range_max_depth):
    return_dict = defaultdict(int)
    
    for depth in range_max_depth:
        randomforest = RandomForestClassifier(n_estimators=n_estimators_rf,max_depth=depth,oob_score=True,random_state=42)
        randomforest.fit(X_tr, y_train)
        y_pred = (randomforest.predict_proba(X_v)[:,1]>threshold_default)*1
        return_dict['depth: '+str(depth)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(depth)
    
    return return_dict
#%%
#tune_randomforest_max_depth(range(1,11,1))
# result: 6
max_depth_rf = 6
#%%
# tune parameter [prob_threshold]
def tune_randomforest_threshold(range_threshold):
    return_dict = defaultdict(int)
    
    for thresh in range_threshold:
        randomforest = RandomForestClassifier(n_estimators=n_estimators_rf,max_depth=max_depth_rf,oob_score=True,random_state=42)
        randomforest.fit(X_tr, y_train)
        y_pred = (randomforest.predict_proba(X_v)[:,1]>thresh/100)*1
        return_dict['thresh: '+str(thresh/100)] = get_return_dict(y_val,y_pred,df_BorrowerRate_val)
        print(thresh/100)
        
    return return_dict
#%%
#tune_randomforest_threshold(range(90,97,1))
# result: 95%
threshold_rf = 0.95
#%%
# fit model
randomforest = RandomForestClassifier(n_estimators=n_estimators_rf,max_depth=max_depth_rf,oob_score=True,random_state=42)
randomforest.fit(X_tr, y_train)
#%%
y_pred = (randomforest.predict_proba(X_v)[:,1]>threshold_rf)*1
get_return_dict(y_val,y_pred,df_BorrowerRate_val)
#%%
# confusion matrix
print("Random Forest validation confusion matrix with threshold: \n", confusion_matrix(y_val, randomforest.predict_proba(X_v)[:,1]>threshold_rf))
#%%
# feature importance
feature_importances_randomforest = pd.DataFrame(randomforest.feature_importances_,
                                                index = X_orig.columns,
                                                columns=['importance']).sort_values('importance',
                                                        ascending=False)
feature_importances_randomforest[:10]
#%%
# feature importance plot
features = X_orig.columns
importances = randomforest.feature_importances_
indices = np.argsort(importances)[-10:]

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%


#%%
#from datetime import datetime
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'classification_df'+'_'+timestamp
#classification_df.to_csv(r'data/'+filename+'.csv')
#%%
#import sys
#sys.setrecursionlimit(100000)
#
#import pickle
#from datetime import datetime
#
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'classification_df'+'_'+timestamp
#with open('data/'+filename+'.pickle', 'wb') as to_write:
#    pickle.dump(classification_df, to_write)
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
#%%
#%%
#%%
#%%
#%%
# tune parameter [n_estimators]
def tune_randomforest_n_estimators(range_n_estimators):
                    
    """
    Input is range_n_estimators
    Returns dictionary of return dictionaries
    """
    
    return_dict = defaultdict(int)
    
    for n in range_n_estimators:
        randomforest = RandomForestClassifier(n_estimators=n,oob_score=True,random_state=42)
        randomforest.fit(X_tr, y_train)
        
        predict_probas = randomforest.predict_proba(X_v)
        max_index = predict_probas.argmax(axis=1)
        y_pred_list = []
        for i_pred in range(len(max_index)):
            i_class = max_index[i_pred]
            if predict_probas[i_pred][i_class] >= threshold_default:
                y_pred_list.append(randomforest.classes_[i_class])
            else:
                y_pred_list.append(0)
        
        y_pred = np.asarray(y_pred_list)
        return_dict['n: '+str(n)] = get_return_dict(y_val,y_pred,df_returns_val)
        
        print(n)
        
    return return_dict
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
#%%
#%%
#%%
#%%
#%%