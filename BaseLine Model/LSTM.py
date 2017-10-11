from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from keras import backend as K
import numpy as np


seq = Sequential()

seq.add(LSTM(101,input_shape=(15, 101*101), 
				activation='relu', return_sequences=True))
seq.add(BatchNormalization())
model.add(Dropout(0.8))

model.add(LSTM(101, activation='relu'))

seq.add(Dense(1, activation='sigmoid'))

seq.compile(loss='mse', optimizer='adam')

X_train = np.load('DataMat_whole.npy')
label = np.load('LabelMat_whole.npy')
X_train = X_train.reshape(10000, 15, 101*101)

seq.fit(X_train, label, batch_size=2,
epochs=300, validation_split=0.05)

del X_train, label

X_test = np.load('testA.npy')
X_test = X_test.reshape(2000, 15, 101*101)

pred =  seq.predict(X_test)
del X_test
np.save('y_test_LSTM_0.npy', pred)
