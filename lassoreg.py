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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_error
from math import sqrt

import statsmodels.formula.api
import statsmodels.regression.linear_model as sm

from pathlib import Path
import os


from time import time

def train_lassoreg():

	# Importing the dataset
	global X_train, y_train
	BASE_DIR = Path(__file__).resolve().parent
	data_dir = os.path.join(BASE_DIR, 'laso_data')
	X_train = pd.read_csv(f'{data_dir}/xtrain.csv')
	global X_test, y_test
	X_test = pd.read_csv(f'{data_dir}/xtest.csv')

	y_train = X_train['Production']
	y_test = X_test['Production']

	# load selected features
	# we selected the features in the previous lecture / notebook
	# if you haven't done so, go ahead and visit the previous lecture to find out how to select
	# the features

	features = pd.read_csv(f'{BASE_DIR}/FIN_DATA/selected_featuresDaDrLaso.csv', header=None)
	features = [x for x in features[0]]

	# here I will add this last feature, even though it was not selected in our previous step,
	# because it needs key feature engineering steps that I want to discuss further during the deployment
	# part of the course.

	features = features + ['Irrg_No']

	#features
	# reduce the train and test set to the desired features


	X_train = X_train[features]
	X_test = X_test[features]

	lin_model = Lasso(alpha=0.005, random_state=0) # remember to set the random_state / seed
	lin_model.fit(X_train, y_train)


	return lin_model

def test_lassoreg(lin_model):
	t = time()

	pred = lin_model.predict(X_train)
	print('linear train mse: {}'.format(mean_squared_error(np.exp(y_train), np.exp(pred))))
	print('linear train rmse: {}'.format(sqrt(mean_squared_error(np.exp(y_train), np.exp(pred)))))
	#print()
	pred = lin_model.predict(X_test)
	print('linear test mse: {}'.format(mean_squared_error(np.exp(y_test), np.exp(pred))))
	print('linear test rmse: {}'.format(sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))))
	#print()
	print('Average Production: ', np.exp(y_train).median())
	output = lin_model.predict(X_test)

	print("The running time: ",time()-t)

def predict_lassoreg(lin_model, inp):
	t = time()
	#inp = sc.transform(inp)
	medianreg = np.exp(y_train).median()
	output = lin_model.predict(inp)


	print("The running time: ",time()-t)

	return output ,medianreg ,time()-t;
