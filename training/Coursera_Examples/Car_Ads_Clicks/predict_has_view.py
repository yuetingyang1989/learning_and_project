
# coding: utf-8

# In[ ]:

import pandas as pd
from pandas import read_csv
import numpy as np
import random as rnd
import pickle
import datetime as dt

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings('ignore')


# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier



from collections import Counter
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from matplotlib import pyplot


# In[ ]:

__author__ = 'yueting.yang.tue@gmail.com'
APP_NAME = 'rank_predict_car_ads'


# In[ ]:

import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


def text_pre_processing(text):
     #remove punctuation
    text = remove_punctuation(text)
    
    #lowercase & remove extra whitespace
    text = " ".join(text.lower().split()) 
    
    return text


# In[ ]:

def data_cleaning(data):
    
    data['src_ad_id'] = data['src_ad_id'].apply(lambda x: '{:.0f}'.format(x))
    data['group'].fillna('no_test', inplace=True)
    # remove null values 
    data = data[(data['telclicks'].isnull()== False) & (data['price'].isnull()== False)          &(data['model'].isnull()== False) & (data['aantalstoelen'].isnull()== False)]
    
    
    # for emissie is null, it is mostly electrical cars, so we fill in emissie as 0 and energy label as A
    data['energielabel'] = np.where(data['emissie'].isna(), 'A', data['energielabel'])
    data['emissie']= pd.to_numeric(data['emissie'], errors='coerce')
    data['emissie'].fillna(0, inplace=True)
    
    
    data['date'] = pd.to_datetime(data['ad_start_dt'])
    data['day_of_week'] = data['date'].dt.day_name()
    data['days_before_cyber_monday'] = (dt.datetime(2016,11,28) - data['date']).dt.days
    
    
    data['auto_age'] = 2016 - data['bouwjaar']
    data['has_view'] = np.where(((data['telclicks']==0) & (data['bids']==0)                             & (data['n_asq']==0 ) & (data['webclicks']==0)), 0, 1)
    data['nr_view']= data['telclicks'] + data['bids'] + data['n_asq'] + data['webclicks']
    
    
    # test preprocessing for model field to combine the same filed
    data['model_pro']= data['model'].apply(lambda text: text_pre_processing(text))
       
    return data


# In[ ]:

def bin_categorical_feature(data):
    # total number of doors
    col = 'aantaldeuren'
    conditions = [data[col] =='1', (data[col].isin(['2','3'])), (data[col].isin(['4','5'])), (data[col].isin(['6','7','8']))]                                                                     
    choices = ["1door","2_3door","4_5door",'6+door']
    data['nr_door'] = np.select(conditions, choices, default = "Unknown")
    
    # total number of chairs
    col = 'aantalstoelen'
    conditions = [data[col] =='1', (data[col].isin(['2','3'])), (data[col].isin(['4','5'])), (data[col].isin(['6','7','8','9']))]                                                                     
    choices = ["1chair","2_3chair","4_5chair",'6+chair']
    data['nr_chair'] = np.select(conditions, choices, default = "Unknown")

    # photo counts
    col = 'photo_cnt'
    conditions = [data[col] <8, (data[col]>= 8) & (data[col]<= 12),(data[col]>= 13) & (data[col]<= 23)]                                                                     
    choices = ["max_7_foto","8_12_foto","13_23_foto"]
    data['nr_foto'] = np.select(conditions, choices, default = "24_foto")
    data['nr_foto'].value_counts()
    
    # ads post dates
    col = 'days_live'
    conditions = [data[col]==0, data[col]==1, (data[col] >1) & (data[col]<= 7),(data[col] >7) & (data[col]<= 31) ]
    choices = ["today","yesterday","1_week","1_month"]
    data['days_post'] = np.select(conditions, choices, default = "always")
    
    # brand and nations
    col = 'brand'
    conditions = [data[col].isin(['VOLKSWAGEN', 'BMW', 'MERCEDES','OPEL','AUDI','MERCEDES-BENZ'])
                 , data[col].isin(['PEUGEOT','RENAULT','CITROEN'])
                 , data[col].isin(['FORD','CHEVROLET','CHRYSLER'])
                 , data[col].isin(['VOLVO','SAAB'])
                 , data[col].isin(['TOYOTA','NISSAN','SUZUKI','MAZDA','MITSUBISHI','HONDA','DAIHATSU'])
                 , data[col].isin(['SEAT'])
                 , data[col].isin(['HYUNDAI','KIA','DAEWOO'])
                 , data[col].isin(['SKODA'])
                 , data[col].isin(['MINI','LAND ROVER'])
                 ]
    choices = ["german","french","american","swedish","japanese","spanish","korean","czech","british"]
    data['brand_nation'] = np.select(conditions, choices, default = "unknown")
    data['brand'] = np.where(data['brand']=='MERCEDES-BENZ', 'MERCEDES', data['brand'])
    
    # l2 is related to auto driving option
    data['l2_modified'] = np.where(data['l2']=='None', 0, data['l2'])
    data['has_auto_driving'] = np.where(data['l2']=='None', 0, 1)
    data['l2_modified']= pd.to_numeric(data['l2_modified'], errors='coerce')
    
    # engine power
    col = 'vermogen'
    conditions = [data[col]<75, (data[col] >75) & (data[col]<= 100)
                  ,(data[col] >100) & (data[col]<= 125) ,(data[col] >125) & (data[col]<= 150) 
                 ,(data[col] >150) & (data[col]<= 200)]
    choices = ["<75pk","75-100pk","101-125pk","126-150pk","151-200pk"]
    data['vermogen_group'] = np.select(conditions, choices, default = ">200pk")
    
    
    
    # Assumption: kmstand is related to bouwjaar and the size of the car ( imagine the size of the car represent the usage frequency)
    data['has_kmstand'] = np.where(data['kmstand']>0, 1, 0)
    data['kmstand_pro'] = data.groupby(['bouwjaar', 'nr_door'])['kmstand'].transform(lambda x: x.fillna(x.median()))
    data['new_auto']= np.where((data['kmstand']<100) & (data['kmstand']>0), 1, 0)
    
    
    
    return data


# In[ ]:

def train_test_data_split(df, ratio = 0.8)
    msk = np.random.rand(len(df)) < ratio
    train_df = df[msk]
    test_df = df[~msk]
    
    return train_df, test_df


# In[ ]:

import category_encoders as ce
def leave_one_out_encoding(df, cols, target_col):

    col_encoder = ce.LeaveOneOutEncoder(cols =cols)
    col_encoder.fit(df[cols], df[target_col])
    
    return col_encoder


# In[ ]:

def train_test_array(train_df, test_df, target_col):
    X_train = train_df.drop([target_col], axis=1).values
    y_train = train_df[target_col].values

    X_test = test_df.drop([target_col], axis=1).values
    y_test = test_df[target_col].values
    
    return X_train, X_test, y_train, y_test


# In[ ]:

from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn import discriminant_analysis
from xgboost import XGBClassifier
import lightgbm as lgb

def model_selection(X_train, X_test, y_train, y_test):
    MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    
    #Naives Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    

    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier(),
    
    #LightGBM
    lgb.LGBMClassifier()
    ]
    
    MLA_columns = ['MLA Name', 'MLA Parameters','Train ROC AUC','ROC AUC', 'F1-Score' ]
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    #index through MLA and save performance to table
    row_index = 0
    for alg in MLA:

        #set name and parameters
        MLA_name = alg.__class__.__name__
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

        alg.fit(X_train, y_train)
        y_train_pred = alg.predict(X_train)
        y_pred = alg.predict(X_test)

        roc_auc_train = roc_auc_score(y_train, y_train_pred)   
        MLA_compare.loc[row_index, 'Train ROC AUC'] = roc_auc_train

        roc_auc = roc_auc_score(y_test, y_pred)   
        MLA_compare.loc[row_index, 'ROC AUC'] = roc_auc

        f1 = f1_score(y_test, y_pred, average='weighted')
        MLA_compare.loc[row_index, 'F1-Score'] = f1

        row_index+=1
    
    MLA_compare.sort_values(by = ['F1-Score'], ascending = False, inplace = True)

    return MLA_compare


# In[ ]:

def model_training(best_model_name, X_train, X_test, y_train, y_test):
    
    for alg in MLA:
    #set name and parameters
    MLA_name = alg.__class__.__name__
    if MLA_name == best_model_name
        model = alg
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


# In[ ]:

def evaluation(y_test, y_pred)
    score = f1_score(y_test, y_pred, average='weighted')
    return score


# In[ ]:

def main():
    print('Starting')
    print('Reading data from csv file')
    data = pd.read_csv("Car_dataset.csv") 
    
    print('Data Cleaning')
    data = data_cleaning(data)
    
    print('Select data from group B')
    df = data[data['group']== 'B']
    target_list = ['src_ad_id', 'telclicks', 'bids','n_asq','webclicks',
                   'group','l2','ad_start_dt','date','nr_view','model'
                   ,'bouwjaar','aantaldeuren','aantalstoelen','photo_cnt','kmstand']

    df = df.loc[:, ~df.columns.isin(target_list)]

    print('Feature Engineering')
    print('Feature Engineering step 1: Bining categorical features')
    df = bin_categorical_feature(df)
    
    
    print('Feature Engineering step 2: Leave one out categorical features')
    train_df, test_df =  train_test_data_split(df, 0.8)
    cols = ['kleur','carrosserie','energielabel', 'brand', 'day_of_week'
            ,'nr_door', 'nr_chair', 'nr_foto', 'brand_nation','days_post', 'vermogen_group','model_pro']
    target_col = 'has_view'
    encoder = leave_one_out_encoding(train_df, cols, target_col)

    train_df[cols]= encoder.transform(train_df[cols])
    test_df[cols] = encoder.transform(test_df[cols])

    X_train, X_test, y_train, y_test = train_test_array(train_df, test_df, target_col)
    
    
    print('Feature Engineering Finished.')
    print('Model Selection')
    MLA_compare = model_selection(X_train, X_test, y_train, y_test)
    best_model_name = MLA_compare.iloc[0]['MLA Name']
    print('Best model is: ',best_model_name)
    print('Best model  ROC AUC %.3f' % MLA_compare.iloc[0]['ROC AUC'])
    print('Best model  F1-Score %.3f' % MLA_compare.iloc[0]['F1-Score'])
    
   
    print('Model Prediction & Evaluation')
    y_pred = model_training(best_model_name, X_train, X_test, y_train, y_test)
    test_df['prediction'] = y_pred
    score = evaluation(y_test, y_pred)

    # save the model to disk
    filename = 'lgb_model.sav'
    pickle.dump(cl, open(filename, 'wb'))
    test_df.to_csv('prediction.csv')
    print('Model Prediction Finished and Saved')

