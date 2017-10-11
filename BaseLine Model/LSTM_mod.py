from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np


seq = Sequential()

seq.add(LSTM(5000,input_dim=101*101,input_length=15,return_sequences=True))
#seq.add(LSTM(128,input_shape=(15, 101*101), return_sequences=True))
# seq.add(Dense(64))
seq.add(Dropout(0.2))

seq.add(LSTM(2500, return_sequences=True))
# seq.add(Dense(64))

seq.add(LSTM(2500, return_sequences=True))
seq.add(Dropout(0.2))

seq.add(LSTM(2500, return_sequences=True))

seq.add(LSTM(1000))
seq.add(Dense(1), activation='linear')

seq.compile(loss='mean_squared_error', optimizer='adam')

print "loading data"

X_train = np.load('../Data/DataMat_whole_compress.npy')
label = np.load('../Data/LabelMat_whole.npy')
X_train = X_train.reshape(10000,15,101*101)

print "loading complete, fitting begins"
seq.fit(X_train, label, batch_size=100,
epochs=200, validation_split=0.2)

del X_train, label

print "fitting complete, prediction begins"

X_test = np.load('../Data/test_whole_compress.npy')
X_test = X_test.reshape(2000,15,101*101)

pred =  seq.predict(X_test)
del X_test
np.save('y_test_LSTM_1.npy', pred)
