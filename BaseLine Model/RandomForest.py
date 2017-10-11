import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.metrics import make_scorer
from sklearn import metrics

def RMSE(real, predict):
	 return float(np.sqrt(metrics.mean_squared_error(real, predict)))

def search(X,y):
	rmse = make_scorer(RMSE, greater_is_better = False)

	param_test1 = {'n_estimators':range(150,401,50)}
	gsearch1 = GridSearchCV(estimator = RandomForestRegressor(min_samples_split=30,
                           	min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), 
                      		param_grid = param_test1, scoring=rmse,cv=5)
	gsearch1.fit(X,y)
	print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

def crossV(model, X, y, folds = 5):

	rmse = make_scorer(RMSE, greater_is_better = False)

	scores = cross_val_score(model, X, y, cv = folds, scoring=rmse, n_jobs = 1)

	print scores
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	
def prediction(model):
	print "predicting the data"
	X_test1 = np.load('test_whole_only_H1_aug.npy')
	X_test1 = X_test1.reshape(2000,-1)
	y_test1 = model.predict(X_test1)
	del X_test1
	pred = y_test1
	np.save('y_test_3.npy', y_test1)


def main():

	print "loading data..."

	X_train = np.load('DataMat_whole_only_H1_aug.npy')
	label = np.load('label_whole_only_H1_aug.npy')
	X_train = X_train.reshape(label.shape[0],-1)

	# search(X_train, label)

	
	print "Training the model..."
	model = RandomForestRegressor(
									n_estimators = 500,
									oob_score = True,
									min_samples_split = 100,
									min_samples_leaf = 20,
									random_state = 10,
									max_depth =  3,
									max_features = 'sqrt',
									n_jobs = -1
									)

	# print "Cross Validation"
	# crossV(model, X_train, label, folds=5)
	
	model = model.fit(X_train, label)
	del X_train
	del label
	
	# print model.oob_score_
	# print model.feature_importances_

	print "Prediction"
	prediction(model)

	

	
if __name__ == '__main__':
	main()

