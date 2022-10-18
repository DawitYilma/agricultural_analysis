import numpy as np
import pandas as pd

# for plotting
import matplotlib.pyplot as plt

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import mean_squared_error
from math import sqrt

from pathlib import Path
import os

import statsmodels.formula.api
import statsmodels.regression.linear_model as sm


from time import time

def train_logreg():

	# Importing the dataset
	global X_train, y_train
	BASE_DIR = Path(__file__).resolve().parent
	data_dir = os.path.join(BASE_DIR, 'logreg_data')
	X_train = pd.read_csv(f'{data_dir}/xtrain.csv')
	global X_test, y_test
	X_test = pd.read_csv(f'{data_dir}/xtest.csv')

	y_train = X_train['CropLabel']
	y_test = X_test['CropLabel']

	# drop unnecessary variables from our training and testing sets
	X_train.drop(['CropLabel'], axis=1, inplace=True)
	X_test.drop(['CropLabel'], axis=1, inplace=True)

	X_train = X_train.iloc[:, 2:]
	X_test = X_test.iloc[:, 2:]
	# load selected features
	# we selected the features in the previous lecture / notebook
	# if you haven't done so, go ahead and visit the previous lecture to find out how to select
	# the features

	features = pd.read_csv(f'{BASE_DIR}/FIN_DATA/selected_featuresDaDrLog.csv', header=None)
	features = [x for x in features[0]]

	# here I will add this last feature, even though it was not selected in our previous step,
	# because it needs key feature engineering steps that I want to discuss further during the deployment
	# part of the course.

	features = features + ['Irrg_No']

	#features
	# reduce the train and test set to the desired features


	X_train = X_train[features]
	X_test = X_test[features]


	classifier2=LogisticRegression(random_state=0)
	classifier2.fit(X_train,y_train)


	return classifier2

def test_logreg(classifier2):
	t = time()

	pred = classifier2.predict(X_train)
	print('linear train mse: {}'.format(mean_squared_error(np.exp(y_train), np.exp(pred))))
	print('linear train rmse: {}'.format(sqrt(mean_squared_error(np.exp(y_train), np.exp(pred)))))
	#print()
	pred = classifier2.predict(X_test)
	print('linear test mse: {}'.format(mean_squared_error(np.exp(y_test), np.exp(pred))))
	print('linear test rmse: {}'.format(sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))))
	#print()
	print('Average Production: ', np.exp(y_train).median())
	output = classifier2.predict(X_test)

	print("The running time: ",time()-t)

def predict_logreg(classifier2, inp):
	t = time()
	#inp = sc.transform(inp)
	medianreg = np.exp(y_train).median()
	output = classifier2.predict(inp)


	print("The running time: ",time()-t)

	return output, time()-t;
