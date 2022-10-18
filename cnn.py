import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# navigate folders
from glob import glob
import os

# to open the images
import cv2

# data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# evaluate model and separate train and test
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# for the convolutional network
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing import image
from keras.utils import np_utils

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model

import config

from time import time

def train_cnn():

	# here are all our images
	DATA_FOLDER = 'v2-plant-seedlings-dataset'

	# let's create a dataframe:
	# the dataframe stores the path to the image in one column
	# and the class of the weed (the target) in the next column


	images_df = []

	# navigate within each folder
	for class_folder_name in os.listdir(DATA_FOLDER):
    		class_folder_path = os.path.join(DATA_FOLDER, class_folder_name)

    		# collect every image path
    		for image_path in glob(os.path.join(class_folder_path, "*.png")):

        		tmp = pd.DataFrame([image_path, class_folder_name]).T
        		images_df.append(tmp)

	# concatenate the final df
	images_df = pd.concat(images_df, axis=0, ignore_index=True)
	images_df.columns = ['image', 'target']

	# train_test_split

	X_train, X_test, y_train, y_test = train_test_split(images_df['image'], images_df['target'], test_size=0.20, random_state=101)

	# reset index, because later we iterate over row number

	X_train.reset_index(drop=True, inplace=True)
	X_test.reset_index(drop=True, inplace=True)

	# reset index in target as well
	y_train.reset_index(drop=True, inplace=True)
	y_test.reset_index(drop=True, inplace=True)

	# let's prepare the target
	# it is a multiclass classification, so we need to make
	# one hot encoding of the target

	encoder = LabelEncoder()
	encoder.fit(y_train)

	train_y = np_utils.to_categorical(encoder.transform(y_train))
	test_y = np_utils.to_categorical(encoder.transform(y_test))

	global IMAGE_SIZE
	IMAGE_SIZE = 150

	def im_resize(df, n):
    		im = cv2.imread(df[n])
    		im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    		#print(im)
    		return im

	tmp = im_resize(X_train, 7)

    # the shape of the datasets needs to be (n1, n2, n3, n4)
	# where n1 is the number of observations
	# n2 and n3 are image width and length
	# and n4 indicates that it is a color image, so 3 planes per image

	def create_dataset(df, image_size):
    		# functions creates dataset as required for cnn
    		tmp = np.zeros((len(df), image_size, image_size, 3), dtype='float32')

    		for n in range(0, len(df)):
       			im = im_resize(df, n)
        		tmp[n] = im

    		print('Dataset Images shape: {} size: {:,}'.format(tmp.shape, tmp.size))
    		return tmp

	global x_train, x_test
	x_train = create_dataset(X_train, IMAGE_SIZE)
	x_test = create_dataset(X_test, IMAGE_SIZE)

    		# Source: https://www.kaggle.com/fmarazzi/baseline-keras-cnn-roc-fast-5min-0-8253-lb

			# this is our cnn

	kernel_size = (3,3)
	pool_size= (2,2)
	first_filters = 32
	second_filters = 64
	third_filters = 128

	dropout_conv = 0.3
	dropout_dense = 0.3

	model = Sequential()
	model.add(Conv2D(first_filters, kernel_size, activation = 'relu',
	                 input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
	model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
	#model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
	model.add(MaxPooling2D(pool_size = pool_size))
	model.add(Dropout(dropout_conv))

	model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
	model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
	#model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
	model.add(MaxPooling2D(pool_size = pool_size))
	model.add(Dropout(dropout_conv))

	model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
	model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
	#model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
	model.add(MaxPooling2D(pool_size = pool_size))
	model.add(Dropout(dropout_conv))

	model.add(Flatten())
	model.add(Dense(256, activation = "relu"))
	model.add(Dropout(dropout_dense))
	model.add(Dense(2, activation = "softmax"))

	model.compile(Adam(lr=0.0001), loss='binary_crossentropy',
              metrics=['accuracy'])

	global batch_size, epochs
	batch_size = 10
	epochs = 1

	filepath = "model.h5"
	checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1,
                             save_best_only=True, mode='max')

	reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=1,
                                   verbose=1, mode='max', min_lr=0.00001)


	callbacks_list = [checkpoint, reduce_lr]

	history = model.fit(x=x_train, y=train_y,
                    batch_size=batch_size,
                    validation_split=10,
                    epochs=epochs,
                    verbose=2,
                    callbacks=callbacks_list)


	return model

def test_cnn(model):
	t = time()
	# make a prediction

	output = model.predict_classes(x_test, verbose=1)

	print("The running time: ",time()-t)

def load_pipeline_keras() -> Pipeline:
	m = train_cnn()
	dataset = joblib.load(config.PIPELINE_PATH)

	build_model = lambda: load_model(config.MODEL_PATH)

	model = KerasClassifier(build_fn=build_model,
                                 batch_size = 10,
                                 validation_split=10,
                                 epochs=1,
                                 verbose=2,
                                 callbacks=m.callbacks_list,
                                 #image_size = config.IMAGE_SIZE
                                 )
	model.classes_ = joblib.load(config.ENCODER_PATH)
	model.model = build_model()

	return Pipeline([
        ('dataset', dataset),
        ('cnn_model', model)
    ])


def load_single_image(data_folder: str, filename: str) -> pd.DataFrame:
    """Makes dataframe with image path and target."""

    image_df = []

    # search for specific image in directory
    for image_path in glob(os.path.join(data_folder, f'{filename}')):
        tmp = pd.DataFrame([image_path, 'unknown']).T
        image_df.append(tmp)

    # concatenate the final df
    images_df = pd.concat(image_df, axis=0, ignore_index=True)
    images_df.columns = ['image', 'target']

    return images_df

global KERAS_PIPELINE
KERAS_PIPELINE = load_pipeline_keras()

def make_single_prediction(image_name, image_directory, model):
    """Make a single prediction using the saved model pipeline.

        Args:
            image_name: Filename of the image to classify
            image_directory: Location of the image to classify

        Returns
            Dictionary with both raw predictions and readable values.
        """


	#def im_resize(df, n):
		#im = cv2.imread(df[n])
		#im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
		#return im

	#def create_dataset(df, image_size):
		# functions creates dataset as required for cnn
		#tmp = np.zeros((len(df), image_size, image_size, 3), dtype='float32')


		#for n in range(0, len(df)):
			#im = im_resize(df, n)
			#tmp[n] = im

		#print('Dataset Images shape: {} size: {:,}'.format(tmp.shape, tmp.size))
		#return tmp


    image_df = load_single_image(
        data_folder=image_directory,
        filename=image_name)

    prepared_df = image_df['image'].reset_index(drop=True)
	#prepared_df = create_dataset(prepared_df, IMAGE_SIZE)

    output = KERAS_PIPELINE.predict(prepared_df, verbose=1)

    readable_predictions = ENCODER.encoder.inverse_transform(predictions)

    #_logger.info(f'Made prediction: {predictions}'
                # f' with model version: {_version}')

    return output , readable_predictions


def predict_cnn(model, inp):

	t = time()

	def im_resize(df, n):
		im = cv2.imread(df[n])
		im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
		return im

	def create_dataset(df, image_size):
		# functions creates dataset as required for cnn
		tmp = np.zeros((len(df), image_size, image_size, 3), dtype='float32')


		for n in range(0, len(df)):
			im = im_resize(df, n)
			tmp[n] = im

		print('Dataset Images shape: {} size: {:,}'.format(tmp.shape, tmp.size))
		return tmp

	#inp = create_dataset(inp, IMAGE_SIZE)
	output = model.predict_classes(inp, verbose=1)

	print("The running time: ",time()-t)

	return output, time()-t;
