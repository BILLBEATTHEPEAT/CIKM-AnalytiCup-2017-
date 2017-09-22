from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
import numpy as np
from sklearn import preprocessing


seq = Sequential()
seq.add(ConvLSTM2D(filters=101, kernel_size=(5, 5),
                   input_shape=(None, 101, 101, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=101, kernel_size=(5, 5),
                   padding='same'))
seq.add(BatchNormalization())
seq.add(Dropout(0.2))

seq.add(Flatten())
seq.add(Dense(64, activation='relu'))
# seq.add(ConvLSTM2D(filters=50, kernel_size=(3, 3),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(ConvLSTM2D(filters=1, kernel_size=(3, 3),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

seq.add(Dense(1, activation='tanh'))

seq.compile(loss='mse', optimizer='adam')

scaler = preprocessing.StandardScaler()

X_train = np.load('DataMat_whole_only_H1_aug.npy')
label = np.load('label_whole_only_H1_aug.npy')
# X_train = X_train[::,::, 25:76, 25:76].reshape(10000,15,51,51,1)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train = X_train.reshape(30000,10,101,101,1)

# label = label.reshape(10000,1,1,1,1)

seq.fit(X_train, label, batch_size=5, epochs=100, validation_split=0.1)

del X_train, label

X_test = np.load('test_whole_only_H1_aug.npy')
# X_test = X_test[::,::, 25:76, 25:76].reshape(2000,15,51,51,1)
X_test = scaler.transform(X_test)
X_test = X_test.reshape(2000,10,101,101,1)


pred =  seq.predict(X_test)
del X_test
np.save('y_test_convLSTM_0.npy', pred)
