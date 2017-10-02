import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

print "loading data"

data = np.load('DataMat_whole_only_H1_last10_aug_10.npy')
Y = np.load('label_whole_only_H1_aug_last10_10.npy')
X_train = data[:90000,:]
label_train = Y[:90000,:]
X_eval = data[90000:,:]
label_eval = Y[90000:,:]
del data, Y

lgb_train = lgb.Dataset(X_train, label_train)
lgb_eval = lgb.Dataset(X_eval, label_eval)

print "loading complete"

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
	}

print "training the model"

model = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
				early_stopping_rounds=1000)

del X_train,label_train, X_test, label_test

print "saving model"
model.save_model('light_model.txt')

print 'predicting'

X_test = np.load('test_whole_only_H1_last10_aug.npy')
X_test = X_test.reshape(2000,10,101,101,1)
X_test = X_test[::,::, 25:76, 25:76].reshape(2000,10,51,51,1)
X_test = X_test.reshape(2000,-1)

pred = model.predict(X_test, num_iteration=model.best_iteration)
del X_test
np.save('y_test_LGBM_0.npy', pred)
