from flask import Flask, render_template, request, send_file

from database import app, db, Polynomial, RandomForst

from polyreg import train_polyreg, test_polyreg, predict_polyreg
from randforst import train_randforst, test_randforst, predict_randforst
from logreg import train_logreg, test_logreg, predict_logreg
#from cnn import train_cnn, test_cnn, predict_cnn, make_single_prediction
from multilinear import train_multilinear, test_multilinear, predict_multilinear
from lassoreg import train_lassoreg, test_lassoreg, predict_lassoreg
from svm import train_svm, test_svm, predict_svm

#from config2 import get_logger, UPLOAD_FOLDER

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel

from time import time


#app = Flask(__name__)
#app.url_map.strict_slashes = False

@app.route('/')
def index():
	return render_template('index.html')


@app.route("/download-file/")
def polydownload():
    return send_file('data1.csv', attachment_filename='data1.csv', as_attachment=True)


@app.route('/Poly_data')
def Poly_data():

	polyall = Polynomial.query.all()
	lengp = len(polyall)
	data1 = pd.DataFrame(columns = ['Irrg_No1 Seed_Not_Improved Damage_Yes Dreason_Insects Dreason_Toolittle_rain Dreason_Toomuch_rain Dmeasure_Yes Dmtype_NonChemical Dmchem_Fungicide Fert_Yes Fert_No Ferttype_Natural Ferttype_Chemical Ferttype_Both D22a_DAP D22a_Urea_DAP D22a_Urea_NPS D23_Manure Area Yield Irrg_No'])
	#data1 = pd.DataFrame(columns = ['A'])

	for row in range(lengp):

		#da = polyall.split()
		data1.loc[len(data1)] = polyall[row]

	filename='data1.csv'
	data1.to_csv(filename, index = None)
	#poly_csv = pd.DataFrame
	#poly_csv = polyall.to_csv('data.csv')

	return render_template('poly1.html', polyall = polyall, lengp = lengp, btn = 'polydownload.html')

@app.route('/Polynomial_model')
def Polynomial_model():
	return render_template('FIN_DRAFT_POLY_REG.html')

@app.route('/Polynomial_reg')
def Polynomial_reg():
	return render_template('Polynomial_reg.html')

@app.route('/predict_polyreg', methods=['POST'])
def poly_reg_user():

	if(request.form['space']=='None'):
		data = []
		string = 'value'
		for i in range(1,20):
			data.append(float(request.form['value'+str(i)]))

		for i in range(19):
			print(data[i])

	else:
		string = request.form['space']
		data = string.split()
		print(data)
		print("Type:", type(data))
		print("Length:", len(data))
		for i in range(19):
			print(data[i])
		data = [float(x.strip()) for x in data]

		for i in range(19):
			print(data[i])
	print(data[0])
	poly = Polynomial(data[0],	data[1],	data[2],	data[3],	data[4],	data[5],	data[6],	data[7],	data[8],	data[9],	data[10],	data[11],	data[12],	data[13],	data[14],	data[15],	data[16],	data[17],	data[18],	data[19],	data[20])

	db.session.add(poly)
	db.session.commit()

	print(poly.id)

	data_np = np.asarray(data, dtype = float)
	print(data_np)
	data_np = data_np.reshape(1,-1)
	print(data_np)
	output,medianreg, t = predict_polyreg(lin_reg, data_np)


	return render_template('result_poly_reg.html', output= np.exp(output), medianreg = medianreg, time=t)

@app.route("/download-file-rand/")
def randdownload():
    return send_file('data2.csv', attachment_filename='data2.csv', as_attachment=True)

@app.route('/Rand_data')
def Rand_data():

	randall = RandomForst.query.all()
	lengp = len(randall)
	data2 = pd.DataFrame(columns = ['Max Temprature Humidity D22a_DAP D22a_Urea_DAP Dreason_Insects D22a_Urea_NPS D23_Manure Area Produciton Irrg_No'])
	#data2 = pd.DataFrame(columns = ['A'])

	for row in range(lengp):

		#da = polyall.split()
		data2.loc[len(data2)] = randall[row]

	filename='data2.csv'
	data2.to_csv(filename, index = None)

	return render_template('rand.html', randall = randall, lengp = lengp, btn = 'randdownload.html')


@app.route('/Random_Forst_Class_model')
def Random_Forst_Class_model():
	return render_template('FIN_DRAFT_RANDFORST_MODEL.html')

@app.route('/Random_Forst_Class')
def Random_Forst_Class():
	return render_template('Random_Forst_Class.html')

@app.route('/predict_random_forst', methods=['POST'])
def randforst_user():

	if(request.form['space']=='None'):
		data = []
		string = 'value'
		for i in range(1,10):
			data.append(float(request.form['value'+str(i)]))

		for i in range(9):
			print(data[i])

	else:
		string = request.form['space']
		data = string.split()
		print(data)
		print("Type:", type(data))
		print("Length:", len(data))
		for i in range(9):
			print(data[i])
		data = [float(x.strip()) for x in data]

		for i in range(9):
			print(data[i])

	rand = RandomForst(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9])
	db.session.add(rand)
	db.session.commit()

	print(rand.id)
	randall = RandomForst.query.all()

	data_np = np.asarray(data, dtype = float)
	data_np = data_np.reshape(1,-1)
	out, t = predict_randforst(classifier, data_np)

	if(out==0):
		output = 'Very Low Productivity'
	elif(out==1):
		output = 'Low Productivity'
	elif(out==2):
		output = 'Average Productivity'
	elif(out==3):
		output = 'Good Productivity'
	else:
		output = 'High Productivity'

	return render_template('result_randforst.html', output=output, time=t)


@app.route('/Logistic_modle')
def Logistic_modle():
	return render_template('FIN_DRAFT_LOG_MODEL.html')

@app.route('/Logistic_Reg')
def Logistic_Reg():
	return render_template('Logistic_Reg.html')

@app.route('/predict_logisticreg', methods=['POST'])
def Logistic_Reg_user():

	if(request.form['space']=='None'):
		data = []
		string = 'value'
		for i in range(1,29):
			data.append(float(request.form['value'+str(i)]))

		for i in range(28):
			print(data[i])

	else:
		string = request.form['space']
		data = string.split()
		print(data)
		print("Type:", type(data))
		print("Length:", len(data))
		for i in range(28):
			print(data[i])
		data = [float(x.strip()) for x in data]

		for i in range(28):
			print(data[i])

	data_np = np.asarray(data, dtype = float)
	data_np = data_np.reshape(1,-1)
	out, t = predict_randforst(classifier2, data_np)

	if(out==0):
		output = 'Very Low Productivity'
	elif(out==1):
		output = 'Low Productivity'
	elif(out==2):
		output = 'Average Productivity'
	elif(out==3):
		output = 'Good Productivity'
	else:
		output = 'High Productivity'

	return render_template('result_logisticreg.html', output=output, time=t)

@app.route('/SVM_Clasif_model')
def SVM_Clasif_model():
	return render_template('FIN_DRAFT_SVM_MODEL.html')

@app.route('/SVM_Clasif')
def SVM_Clasif():
	return render_template('SVM_Clasif.html')

@app.route('/predict_svm', methods=['POST'])
def svm_user():

	if(request.form['space']=='None'):
		data = []
		string = 'value'
		for i in range(1,10):
			data.append(float(request.form['value'+str(i)]))

		for i in range(9):
			print(data[i])

	else:
		string = request.form['space']
		data = string.split()
		print(data)
		print("Type:", type(data))
		print("Length:", len(data))
		for i in range(9):
			print(data[i])
		data = [float(x.strip()) for x in data]

		for i in range(9):
			print(data[i])

	data_np = np.asarray(data, dtype = float)
	data_np = data_np.reshape(1,-1)
	out, t = predict_randforst(classifier3, data_np)

	if(out==0):
		output = 'Very Low Productivity'
	elif(out==1):
		output = 'Low Productivity'
	elif(out==2):
		output = 'Average Productivity'
	elif(out==3):
		output = 'Good Productivity'
	else:
		output = 'High Productivity'

	return render_template('result_svm.html', output=output, time=t)


@app.route('/home3')
def hello_method3():
	return render_template('home3.html')

@app.route('/predict3', methods=['POST'])
def login_user3():
	#image = request.files.get('image', '')
	image = request.files['image']
	filename = secure_filename(image.filename)
	#image = request.form['image']
	#filename = image.filename
	#filepath = os.path.join('C:/Users/hp/Agicultural Analysis/v2-plant-seedlings-dataset/Black-grass', filename)
	file.save(os.path.join(UPLOAD_FOLDER, filename))
	result = make_single_prediction(
                image_name=filename,
                image_directory=UPLOAD_FOLDER)

	#data_np = np.asarray(data, dtype = float)
	#data_np = data_np.reshape(1,-1)
	out, t = result.get('readable_predictions')
	#out, t = make_single_prediction(filepath, image, model)
	#out, t = predict_cnn(model, image)

	if(out==0):
		output = 'Black-grass'
	elif(out==1):
		output = 'Charlock'
	elif(out==2):
		output = 'Cleavers'
	elif(out==3):
		output = 'Common Chickweed'
	elif(out==4):
		output = 'Common wheat'
	elif(out==5):
		output = 'Fat Hen'
	elif(out==6):
		output = 'Loose Silky-bent'
	elif(out==7):
		output = 'Maize'
	elif(out==8):
		output = 'Scentless Mayweed'
	elif(out==9):
		output = 'Shepherds Purse'
	elif(out==10):
		output = 'Small-flowered Cranesbill'
	else:
		output = 'Sugar beet'

	return render_template('result3.html', output=output, time=t)

@app.route('/Multi_Linear_model')
def Multi_Linear_model():
	return render_template('FIN_DRAFT_MULTILINEAR_REG.html')

@app.route('/Multi_Linear')
def Multi_Linear():
	return render_template('Multi_Linear.html')

@app.route('/predict_multilinear', methods=['POST'])
def Multi_Linear_User():

	if(request.form['space']=='None'):
		data = []
		string = 'value'
		for i in range(1,20):
			data.append(float(request.form['value'+str(i)]))

		for i in range(19):
			print(data[i])

	else:
		string = request.form['space']
		data = string.split()
		print(data)
		print("Type:", type(data))
		print("Length:", len(data))
		for i in range(19):
			print(data[i])
		data = [float(x.strip()) for x in data]

		for i in range(19):
			print(data[i])

	data_np = np.asarray(data, dtype = float)
	data_np = data_np.reshape(1,-1)
	output,medianreg, t = predict_multilinear(lin_reg2, data_np)


	return render_template('result_multilinear.html', output= np.exp(output), medianreg = medianreg, time=t)

@app.route('/Lasso_model')
def Lasso_model():
	return render_template('FIN_DRAFT_LASO_MODEL.html')

@app.route('/Lasso_reg')
def Lasso_reg():
	return render_template('Lasso_reg.html')

@app.route('/predict_Lasso_reg', methods=['POST'])
def Lasso_reg_User():

	if(request.form['space']=='None'):
		data = []
		string = 'value'
		for i in range(1,16):
			data.append(float(request.form['value'+str(i)]))

		for i in range(15):
			print(data[i])

	else:
		string = request.form['space']
		data = string.split()
		print(data)
		print("Type:", type(data))
		print("Length:", len(data))
		for i in range(15):
			print(data[i])
		data = [float(x.strip()) for x in data]

		for i in range(15):
			print(data[i])

	data_np = np.asarray(data, dtype = float)
	data_np = data_np.reshape(1,-1)
	output,medianreg, t = predict_lassoreg(lin_model, data_np)


	return render_template('result_Lassor_reg.html', output= np.exp(output), medianreg = medianreg, time=t)




@app.route('/profile')
def display3():
	return render_template('profile.html')




if __name__=='__main__':
	global lin_reg, lin_reg2, classifier, classifier2, classifier3, lin_model

	lin_reg = train_polyreg()
	lin_reg2 = train_multilinear()
	classifier = train_randforst()
	classifier2 = train_logreg()
	classifier3 = train_svm()
	lin_model = train_lassoreg()
	#model = train_cnn()

	test_polyreg(lin_reg)
	test_multilinear(lin_reg2)
	test_randforst(classifier)
	test_logreg(classifier2)
	test_svm(classifier3)
	test_lassoreg(lin_model)
	#test_cnn(model)


	print("Done")
	#app.debug = True
	app.run(port=4995)
