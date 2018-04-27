import os
import time
import numpy as np
import pandas as pd
import sys, re, csv, codecs, string

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb

if __name__ == '__main__':
    train = pd.read_csv('./input/train.csv')
    test = pd.read_csv('./input/test.csv')

    train = train.select_dtypes(include=['float64', 'int64'])
    test  = test.select_dtypes(include=['float64', 'int64'])

    target = ['Survived']
    x_train = train.drop(target, axis=1)
    y_train = train['Survived']

    X_train,X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.1)

    dtrain = lgb.Dataset(X_train, label=Y_train)

    myparams = {
        'objective': 'binary',
        'metric': {'auc'},
        'learning_rate': 0.1,
    }

    model = lgb.train(params=myparams, train_set=dtrain)

    my_pred = model.predict(X_valid)

    auc_score = roc_auc_score(Y_valid, my_pred)
    print (auc_score)
