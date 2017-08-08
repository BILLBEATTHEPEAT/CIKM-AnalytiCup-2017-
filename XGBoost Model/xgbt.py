import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split



def main():
	print "loading data"

	X = np.load('train_only_H1_aug_8_Last4_uint8.npy')
	Y = np.load('label_aug_8.npy')
	# Y = Y[:80000,:]
	print Y.shape
	#X = X.reshape(60000,10,101,101,1)
	# X = X[::,::, 25:76, 25:76,::].reshape(60000,10,51,51,1)
	X = X.reshape(Y.shape[0],-1)
	print X.shape, Y.shape
	print X.shape
	x_train, x_eval, y_train, y_eval = train_test_split(
						X, Y, test_size=0.025, random_state=17)
	del X, Y
	# del X, Y, X_test, y_test

	print "loading complete"

	model = XGBRegressor(
		learning_rate = 0.01,
		n_estimators = 1200,
		max_depth = 15,
		min_child_weight = 3,
		gamma = 0.1,
		reg_lambda = 10,
		subsample = 0.8,
		reg_alpha = 1,
		colsample_bytree = 0.8,
		objective = 'reg:linear',
		nthread = -1,
		silent = False,
		scale_pos_weight = 1)

	# modelfit(model, xgtrain)
	# del xgtrain
	# prediction(model, X_train, Y)
	print "trianing the model"
	model.fit(x_train, y_train, eval_metric='rmse', eval_set=[(x_eval, y_eval)], verbose=True)
	del x_train, x_eval, y_train, y_eval
	# fscore = model.Booster.get_score()


	print "predicting the data"

	X_test = np.load('test_only_H1.npy')
	X_test = X_test.reshape(2000,15,101,101,1)
	X_test = X_test[::,11:15, 25:76, 25:76].reshape(2000,4,51,51,1)
	X_test = X_test.reshape(2000,-1)
	preds = model.predict(X_test)
	del X_test
	# np.save('feature_imoprtance.npy', fscore)
	np.save('pred_xgbt_Nondiff.npy', preds)

if __name__ == "__main__":
	main()