#parse the input data into training and test chunks
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def plot_input_data(df):
	fig, ax = plt.subplots(figsize = (8,6))
	ax.plot(df['date'], df['x1'], c='b', lw=1, label='x1')
	ax.plot(df['date'], df['x2'], c='g', lw=1, label='x2')
	ax.plot(df['date'], df['x3'], c='purple', lw=1, label='x3')
	ax.plot(df['date'], df['target'], c='red', lw=2, label='y')
	#ax.set_xticklabels(df['date'][::50], rotation='vertical')
	ax.legend()
	ax.set_xlabel("Date", fontsize=14)
	ax.set_ylabel("Price", fontsize = 14)
	return(fig)

def transform_input_data(df):
	X = df[['x1','x2','x3']].values
	y = df['target'].values
	X[1:] = (X[1:]/X[:-1])
	X[0] = np.ones(3)
	y[1:] = y[1:]/y[:-1]
	y[0] = 1.0
	X = np.log(X); y = np.log(y)
	return(X, y)

def plot_transformed_data(X,y):
	fig, ax = plt.subplots(figsize = (8,6))
	ax.plot(X[:,0], c='b', lw=1, label='x1')
	ax.plot(X[:,1], c='g', lw=1, label='x2')
	ax.plot(X[:,2], c='purple', lw=1, label='x3')
	ax.plot(y, c='r', lw=2, label='y')
	ax.legend()
	ax.set_xlabel("Day number", fontsize=14)
	ax.set_ylabel("Log daily returns", fontsize = 14)
	return(fig)

def make_train_test(X,y):
	Xtrain = X[:int(X.shape[0]*.8)]
	Xtest = X[int(X.shape[0]*.8):]
	ytrain = y[:int(y.size*.8)]
	ytest = y[int(y.size*.8):]

	trainX = np.zeros((Xtrain.shape[0]-14, 14, 3))
	trainy = np.zeros((ytrain.shape[0]-14))
	for i in range(Xtrain.shape[0]-14):
		trainX[i,:,:] = Xtrain[i:i+14]
		trainy[i] = ytrain[i+14]

	testX = np.zeros((Xtest.shape[0]-14, 14, 3))
	testy = np.zeros((ytest.shape[0]-14))
	for i in range(Xtest.shape[0]-14):
		testX[i,:,:] = Xtest[i:i+14]
		testy[i] = ytest[i+14]
	return(trainX, trainy, testX, testy)



