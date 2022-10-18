import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from time import time

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error
from math import sqrt

from pathlib import Path
import os

def train_randforst():

	# Importing the dataset
	# Importing the dataset
	global X_train, y_train
	BASE_DIR = Path(__file__).resolve().parent
	data_dir = os.path.join(BASE_DIR, 'Randforst_data')
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

	pca = PCA(n_components = 10)
	X_train=pca.fit_transform(X_train)
	X_test=pca.transform(X_test)
	explained_variance=pca.explained_variance_ratio_

	# train the model
	classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
	classifier.fit(X_train,y_train)

	return classifier

def test_randforst(classifier):
	t = time()

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
	print('Average CropLabel: ', np.exp(y_train).median())

	output = classifier.predict(X_test)
	acc = accuracy_score(y_test, output) 
	print("The accuracy of testing data: ",acc)
	print("The running time: ",time()-t)

def predict_randforst(classifier, inp):
	t = time()
	#inp = sc.transform(inp)
	output = classifier.predict(inp)
	
	print("The running time: ",time()-t)

	return output, time()-t;