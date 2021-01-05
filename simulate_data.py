#generate fake data

import numpy as np
import pandas as pd
import datetime

def generate_data(Ndays=500, start = '01/01/2017'):
	auto_cov = np.eye(3)
	auto_cov[0,1] = .2
	auto_cov[0,2] = .4
	auto_cov = (auto_cov + auto_cov.T)/2
	auto_cov *= .01
	X = np.random.multivariate_normal(np.zeros(3), auto_cov, Ndays)
	X[2:] = .1*X[:-2] + .3*X[1:-1] + .6*X[2:]
	X = np.exp(X)
	X = np.cumprod(X, axis=0)
	X = X*np.array([10,30,15])
	df = pd.DataFrame(X, columns = ['x1','x2','x3'])
	y = df[['x1','x2','x3']].ewm(halflife=7).mean().mean(axis=1)
	df['target'] = y
	start = pd.to_datetime(start)
	stop = start + datetime.timedelta(days=Ndays-1)
	df['date'] = pd.date_range(start, stop)
	return(df)

if __name__=='__main__':
	Ndays = 1000
	df = generate_data(Ndays)
	df.to_csv('input_data.csv')

