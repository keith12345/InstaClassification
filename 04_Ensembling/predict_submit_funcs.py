from collections import OrderedDict

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB



 ####################################
 # Training and Test for Submission #
 ####################################

def submit_fit_score_pred_log(train_df, test_df, final_order_id,C=10000):
    
    test_ids = test_df.loc[:,['product_id','user_id']]
    X_test = test_df.drop(['product_id','user_id'],axis=1) 
    
    X_train = train_df.drop(['product_id','user_id',
                          'latest_cart','in_cart'],axis=1) 
    y_train = train_df['in_cart']
    
    lr = LogisticRegression(solver='liblinear',C=C)
    lr.fit(X_train, y_train)
    predictions = pd.DataFrame(lr.predict(X_test))
    
    output = pd.merge(test_ids,predictions,
                      left_index=True,right_index=True)
    
    output = pd.merge(output,final_order_id,
                      on='user_id')
    
    output.columns = ['product_id','user_id',
                      'prediction','order_id']
    
    return output, lr

def submit_fit_score_pred_G_NB(train_df, test_df, final_order_id):
    
    test_ids = test_df.loc[:,['product_id','user_id']]
    X_test = test_df.drop(['product_id','user_id'],axis=1) 
    
    X_train = train_df.drop(['product_id','user_id',
                          'latest_cart','in_cart'],axis=1) 
    y_train = train_df['in_cart']
    
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    predictions = pd.DataFrame(gnb.predict(X_test))
    
    output = pd.merge(test_ids,predictions,
                      left_index=True,right_index=True)
    
    output = pd.merge(output,final_order_id,
                      on='user_id')
    
    output.columns = ['product_id','user_id',
                      'prediction','order_id']
    
    return output, gnb


def submit_fit_score_pred_rfc(train_df, test_df, final_order_id):
    
    test_ids = test_df.loc[:,['product_id','user_id']]
    X_test = test_df.drop(['product_id','user_id'],axis=1) 
    
    X_train = train_df.drop(['product_id','user_id',
                          'latest_cart','in_cart'],axis=1) 
    y_train = train_df['in_cart']
    
    rfc = RandomForestClassifier(n_estimators=10, n_jobs=4)
    rfc.fit(X_train, y_train)
    predictions = pd.DataFrame(rfc.predict(X_test))
    
    output = pd.merge(test_ids,predictions,
                      left_index=True,right_index=True)
    
    output = pd.merge(output,final_order_id,
                      on='user_id')
    
    output.columns = ['product_id','user_id',
                      'prediction','order_id']
    
    return output, rfc


def submit_fit_score_pred_M_NB(train_df, test_df, final_order_id):
    
    train_df.diff_between_average_and_current_order_time = (
        train_df.diff_between_average_and_current_order_time 
        + abs(train_df.diff_between_average_and_current_order_time.min()))
    
    test_df.diff_between_average_and_current_order_time = (
        test_df.diff_between_average_and_current_order_time 
        + abs(test_df.diff_between_average_and_current_order_time.min()))    
    
    test_ids = test_df.loc[:,['product_id','user_id']]
    X_test = test_df.drop(['product_id','user_id'],axis=1) 
    
    X_train = train_df.drop(['product_id','user_id',
                          'latest_cart','in_cart'],axis=1) 
    y_train = train_df['in_cart']
    
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    predictions = pd.DataFrame(mnb.predict(X_test))
    
    output = pd.merge(test_ids,predictions,
                      left_index=True,right_index=True)
    
    output = pd.merge(output,final_order_id,
                      on='user_id')
    
    output.columns = ['product_id','user_id',
                      'prediction','order_id']
    
    return output, mnb



##################################
#   Formatting for Submission    #
##################################

def products_concat(series):
    out = ''
    for product in series:
        if product > 0:
            out = out + str(int(product)) + ' '
    
    if out != '':
        return out.rstrip()
    else:
        return 'None'
    
def format_for_submission(predictions):
    predictions['product_id'] = predictions.prediction * predictions.product_id
    predictions.drop('prediction',axis=1,inplace=True)
    predictions = (pd.DataFrame(predictions.groupby('order_id')
                    ['product_id'].apply(products_concat)).reset_index())
    predictions.columns = ['order_id','products']
    return predictions