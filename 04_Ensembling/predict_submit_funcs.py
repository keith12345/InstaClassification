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

def submit_fit_score_pred_log(df, test_ids):
    """    
    Takes a DataFrame, training, and validation data as its input.
    Returns f1-score, features and their coefficients, and predicted non-re-orders and re-orders.
    """
    
    y = df['in_cart']
    X = df.drop(['product_id','user_id',
                          'latest_cart','in_cart'],axis=1) 
    
    X_pred = X.loc[test_ids.user_id]
    X_pred.reset_index(drop=True,inplace=True)
    test_ids.reset_index(drop=True,inplace=True)
    
    lr = LogisticRegression()
    lr.fit(X, y)
    predictions = pd.DataFrame(lr.predict(X_pred))
    
    output = pd.merge(test_ids,predictions,
                      left_index=True,right_index=True)
    
    return output



#def products_concat(series):
#    out = ''
#    for product in series:
#       if product > 0:
#            out = out + str(int(product)) + ' '
#    
#    if out != '':
#        return out.rstrip()
#    else:
#        return 'None'