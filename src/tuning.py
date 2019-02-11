from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV


def hyperparameter_tuning(mode, scoring, X, Y, save_path=None):
	"""

	:param mode:
	:param scoring: 'roc_auc' for binary
	:param X:
	:param Y:
	:return:
	"""
	n_folds = 5

	if mode == 'knn':
		model = KNeighborsClassifier()
		param_grid = {'n_neighbors': np.arange(3, 50, 2)}

		title = 'KNN Tuning'
		xlabel = 'Number of neighbours, k'
		ylabel = 'Score'
		x_axis = param_grid['n_neighbors']
		xscale = 'linear'

	elif mode == 'svm_linear':
		model = SVC(kernel='linear', max_iter=10000)
		param_grid = {'C': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}

		title = 'SVM Tuning (linear kernel)'
		xlabel = 'C (log scale)'
		ylabel = 'Score'
		x_axis = param_grid['C']
		xscale = 'log'

	elif mode == 'svm_poly':
		model = SVC(kernel='poly', degree=3, max_iter=10000)
		param_grid = {'C': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}

		title = 'SVM Tuning (poly kernel)'
		xlabel = 'C (log scale)'
		ylabel = 'Score'
		x_axis = param_grid['C']
		xscale = 'log'

	elif mode == 'dec_tree':
		model = DecisionTreeClassifier(max_depth=15)
		param_grid = {'max_depth': np.arange(1, 50)}

		title = 'Decision Tree Tuning'
		xlabel = 'Max depth'
		ylabel = 'Score'
		x_axis = param_grid['max_depth']
		xscale = 'linear'

	elif mode == 'boosting':
		model = AdaBoostClassifier(learning_rate=0.1)
		param_grid = {'n_estimators': np.arange(5, 400, 3)}
		# param_grid = {'learning_rate': [1e-2, 1e-1, 1]}

		title = 'Adaboost Tuning'
		xlabel = 'Number of estimators'
		ylabel = 'Score'
		x_axis = param_grid['n_estimators']
		# x_axis = param_grid['learning_rate']
		xscale = 'linear'
		# xscale = 'log'

	elif mode == 'neural_n':
		model = MLPClassifier(max_iter=2000)
		param_grid = {'hidden_layer_sizes': []}
		for n in range(1, 11):
			param_grid['hidden_layer_sizes'].append(tuple(25 for _ in range(n)))

		title = 'Neural Network Tuning (25 hidden units)'
		xlabel = 'Number of layers'
		ylabel = 'Score'
		x_axis = [len(x) for x in param_grid['hidden_layer_sizes']]
		xscale = 'linear'

	else:
		print("Invalid mode entered")
		return None

	# Running the grid search
	grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=n_folds, n_jobs=-1, return_train_score=True)

	start = time()
	grid_search.fit(X, Y)
	print("Time taken for grid search on {}: {}".format(mode, (time()-start)/60))

	# Plot (but not show) the param tuning for this model
	mean_test_score = grid_search.cv_results_['mean_test_score']
	mean_train_score = grid_search.cv_results_['mean_train_score']
	tuning_2d_plot(title, xlabel, ylabel, x_axis, mean_train_score, mean_test_score, xscale=xscale, save_path=save_path)

	# Getting params and training/testing scores to return
	best_param = grid_search.best_params_
	best_ix = np.flatnonzero(grid_search.cv_results_['rank_test_score'] == 1)  # Gets the index with rank 1
	best_test_score = mean_test_score[best_ix]
	best_train_score = mean_train_score[best_ix]

	return best_param, (best_train_score, best_test_score), grid_search.best_estimator_


def tuning_2d_plot(title, xlabel, ylabel, param_list, train_score, test_score, xscale='linear', save_path=None):
	plt.figure()
	plt.plot(param_list, train_score)
	plt.plot(param_list, test_score)
	plt.legend(['Train scores', 'Test scores'])
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xscale(xscale)
	if save_path:
		plt.savefig(save_path)


def test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, model_name, best_param, tuning_scores):

	model.fit(train_X, train_Y)
	pred_Y = model.predict(test_X)
	score = score_fn(test_Y, pred_Y)

	print("=====================================================")
	print("Model: {}".format(model_name))
	print("Best param: {}".format(best_param))
	print("Best train score:      {}".format(tuning_scores[0]))
	print("Best validation score: {}".format(tuning_scores[1]))
	print("Test score:            {}".format(score))