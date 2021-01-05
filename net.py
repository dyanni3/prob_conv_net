#keras conv_prob net

import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.losses import mse
import keras.backend as K


class ConvProbNet:

	def __init__(self, n_days=14):
		self.input_layer = Input((n_days,3))
		self.conv1 = (Conv1D(filters=64, kernel_size=3, 
			activation='relu', input_shape=(n_days, 3)))(self.input_layer)
		self.conv2 = Conv1D(filters=64, kernel_size=3,
		 activation='relu')(self.input_layer)
		self.dropout = Dropout(0.5)(self.conv2)
		self.max_pool = MaxPooling1D(pool_size=2)(self.dropout)
		self.flatten = Flatten()(self.max_pool)
		self.dense = Dense(100, activation='relu')(self.flatten)
		self.mu = Dense(1, activation = 'linear')(self.dense)
		self.sigma2 = Dense(1, activation = 'linear')(self.dense)
		self.y_true = Input((1,))
		self.model = Model(inputs = [self.input_layer, self.y_true],
		 outputs = [self.mu, self.sigma2])
		self.loss = K.mean(mse(self.mu,self.y_true) + mse(K.square(self.mu-self.y_true), self.sigma2))
		self.model.add_loss(self.loss)
		self.model.compile(optimizer='adam', metrics=['accuracy'])

	def train(self, X, y):
		pass

	def eval(self, X, y):
		pass



