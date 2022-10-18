# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 07:24:32 2021

@author: hp
"""

import numpy as np
import pandas as pd

# for plotting
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error

from pathlib import Path
import os

from math import sqrt

BASE_DIR = Path(__file__).resolve().parent
data_dir = os.path.join(BASE_DIR, 'svm_data')

FinDaDr = pd.read_csv(f'{BASE_DIR}/FIN_DATA/Final_Data_Draft_Yield.csv')
FinDaDr.head()

X_train, X_test, y_train, y_test = train_test_split(FinDaDr, FinDaDr.CropLabel,
                                                    test_size=0.2,
                                                    random_state=0) # we are setting the seed here
X_train.shape, X_test.shape

for var in ['PRECIP', 'TMPMAX', 'TMPMIN', 'RELHUM', 'WINDLY', 'SUNHRS', 'Fweight', 'Area', 'Production']:
    X_train[var] = np.log(X_train[var])
    X_test[var]= np.log(X_test[var])
    
train_vars = [var for var in X_train.columns if var not in ['Year', 'CropLabel']]
len(train_vars)

X_train['CropLabel'].reset_index(drop=True)

# fit scaler
scaler = MinMaxScaler() # create an instance
scaler.fit(X_train[train_vars]) #  fit  the scaler to the train set for later use

# transform the train and test set, and add on the Id and SalePrice variables
X_train = pd.concat([X_train[['CropLabel']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(X_train[train_vars]), columns=train_vars)],
                    axis=1)

X_test = pd.concat([X_test[['CropLabel']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(X_test[train_vars]), columns=train_vars)],
                    axis=1)

# capture the target
y_train = X_train['CropLabel']
y_test = X_test['CropLabel']

# drop unnecessary variables from our training and testing sets
X_train.drop(['CropLabel'], axis=1, inplace=True)
X_test.drop(['CropLabel'], axis=1, inplace=True)

X_train = X_train.iloc[:, 2:]
X_test = X_test.iloc[:, 2:]

pca = PCA(n_components = 35)
X_train=pca.fit_transform(X_train)
X_tes=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_

X_t = pd.DataFrame(columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'I', 'J', 'K', 'L', 'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','AA','BB','CC','DD','EE','FF','GG'])

for row in range(84):
    X_t.loc[len(X_t)] = X_tes[row] 

X_t.to_csv(f'{data_dir}/xtestsel.csv')
# train the model
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

# evaluate the model:
# remember that we log transformed the output  in our feature engineering notebook / lecture.

# In order to get the true performance of the Lasso
# we need to transform both the target and the predictions
# back to the original agriculture values.

# We will evaluate performance using the mean squared error and the
# root of the mean squared error

pred = classifier.predict(X_train)
print('linear train mse: {}'.format(mean_squared_error(np.exp(y_train), np.exp(pred))))
print('linear train rmse: {}'.format(sqrt(mean_squared_error(np.exp(y_train), np.exp(pred)))))
print()
pred = classifier.predict(X_test)
print('linear test mse: {}'.format(mean_squared_error(np.exp(y_test), np.exp(pred))))
print('linear test rmse: {}'.format(sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))))
print()
print('Average house price: ', np.exp(y_train).median())

cm=confusion_matrix(y_test, pred)
