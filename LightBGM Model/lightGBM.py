import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


def main():
	print "loading data"

	data = np.load('DataMat_whole_only_H1_last10_aug_10.npy')
	Y = np.load('label_whole_only_H1_aug_last10_10.npy')
	X_train = data[:90000,:]
	label_train = Y[:90000,:].reshape(-1,)
	X_eval = data[90000:,:]
	label_eval = Y[90000:,:].reshape(-1,)
	# print X_train.shape, X_eval.shape, label_train.shape, label_eval.shape
	del data, Y

	# lgb_train = lgb.Dataset(X_train, label_train)
	# lgb_eval = lgb.Dataset(X_eval, label_eval)

	print "loading complete"

	model = LGBMRegressor(boosting_type='gbdt', 
		num_leaves=31, 
		max_depth=-1, 
		learning_rate=0.01, 
		n_estimators=1000, 
		max_bin=255, 
		subsample_for_bin=50000, 
		objective=None, 
		min_split_gain=0, 
		min_child_weight=3,
		min_child_samples=10, 
		subsample=1, 
		subsample_freq=1, 
		colsample_bytree=1, 
		reg_alpha=0.1, 
		reg_lambda=0, 
		seed=17,
		silent=False, 
		nthread=-1)

	# modelfit(model, xgtrain)
	# del xgtrain
	# prediction(model, X_train, Y)
	print "trianing the model"
	model.fit(X_train, label_train, 
		eval_metric='rmse',
		eval_set=[(X_eval, label_eval)],
		verbose = True)
	del X_train, label_train, X_eval, label_eval


	print "predicting the data"

	X_test = np.load('test_whole_only_H1_last10_aug.npy')
	X_test = X_test.reshape(2000,10,101,101,1)
	X_test = X_test[::,::, 25:76, 25:76].reshape(2000,10,51,51,1)
	X_test = X_test.reshape(2000,-1)

	pred = model.predict(X_test, num_iteration=model.best_iteration)
	del X_test
	np.save('y_test_LGBM_0.npy', pred)




if __name__ == "__main__":
	main()