import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data_utils
import learning
from tuning import hyperparameter_tuning, test_and_print
from sklearn.metrics import accuracy_score


def aggregate_wine_labels(Y):
	new_Y = Y.copy()
	new_Y[Y >= 6] = 1   # Good wines
	new_Y[Y < 6]  = 0   # Bad wines
	return new_Y


def remove_nan_and_save(filepath_input, filepath_output):
	"""
	reads csv file and removes data instances with empty features,
	then outputs them into a csv file with ',' delimiter

	:param filepath_input: str, filepath of the input csv file
	:param filepath_output: str, filepath of the output csv file after cleaning
	"""
	df = pd.read_csv(filepath_input, sep=';')

	# Removes all rows that have empty cells
	row_no_nan = df.notna().all(axis=1)
	print("Number of rows with empty cells: {}".format(np.sum(~row_no_nan)))
	df = df[row_no_nan]

	df.to_csv(filepath_output, index=False)


def explore_features(filepath, save_figures=False):
	X, Y = data_utils.get_full_set(filepath)
	Y = aggregate_wine_labels(Y)

	if save_figures:
		save_file_path = '../graphs/wine'
	else:
		save_file_path = None

	data_utils.plot_feature_distribution(X, Y, save_file_path)
	data_utils.plot_outcome_distribution(Y, save_file_path)
	plt.show()


def tune(filepath, save_figures=False):
	### Getting and processing the dataset
	test_X, test_Y, train_X, train_Y = data_utils.get_separate_sets(filepath)

	# Normalize features, and aggregate categories of wine
	#   score >=6 is good wine, labelled 1
	#   score <6 is bad wine, labelled 0
	train_X = data_utils.normalize(train_X)
	train_Y = aggregate_wine_labels(train_Y)
	test_X = data_utils.normalize(test_X)
	test_Y = aggregate_wine_labels(test_Y)


	### Running the tuning
	score_fn = accuracy_score
	score_fn_name = 'accuracy'

	print("Running KNN")
	if save_figures:
		save_path = '../graphs/wine/tuning_knn.png'
	else:
		save_path = None
	best_param, tuning_scores, model = hyperparameter_tuning('knn', score_fn_name, train_X, train_Y, save_path=save_path)
	test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, 'K-Nearest Neighbour', best_param, tuning_scores)

	print("Running SVM (linear)")
	if save_figures:
		save_path = '../graphs/wine/tuning_svm_linear.png'
	else:
		save_path = None
	best_param, tuning_scores, model = hyperparameter_tuning('svm_linear', score_fn_name, train_X, train_Y, save_path=save_path)
	test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, 'Support Vector Machines (linear)', best_param, tuning_scores)

	print("Running SVM (poly)")
	if save_figures:
		save_path = '../graphs/wine/tuning_svm_poly.png'
	else:
		save_path = None
	best_param, tuning_scores, model = hyperparameter_tuning('svm_poly', score_fn_name, train_X, train_Y, save_path=save_path)
	test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, 'Support Vector Machines (poly)', best_param, tuning_scores)

	print("Running decision trees")
	if save_figures:
		save_path = '../graphs/wine/tuning_dt.png'
	else:
		save_path = None
	best_param, tuning_scores, model = hyperparameter_tuning('dec_tree', score_fn_name, train_X, train_Y, save_path=save_path)
	test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, 'Decision Trees', best_param, tuning_scores)

	print("Running boosting")
	if save_figures:
		save_path = '../graphs/wine/tuning_boosting.png'
	else:
		save_path = None
	best_param, tuning_scores, model = hyperparameter_tuning('boosting', score_fn_name, train_X, train_Y, save_path=save_path)
	test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, 'Adaboost', best_param, tuning_scores)

	print("Running neural networks")
	if save_figures:
		save_path = '../graphs/wine/tuning_neuraln.png'
	else:
		save_path = None
	best_param, tuning_scores, model = hyperparameter_tuning('neural_n', score_fn_name, train_X, train_Y, save_path=save_path)
	test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, 'Neural Nets', best_param, tuning_scores)

	plt.show()


def learn(filepath, save_figures=False):
	X, Y = data_utils.get_full_set(filepath)
	X = data_utils.normalize(X)
	Y = aggregate_wine_labels(Y)
	score_fn = accuracy_score

	learning.wine_learning_rate(X, Y, score_fn, save=True)

	plt.show()


if __name__ == '__main__':
	command = sys.argv[1]

	if len(sys.argv) == 3 and sys.argv[2] == 'save':
		save_fig = True
	else:
		save_fig = False

	raw_data_path = '../data/raw-winequality-red.csv'
	data_path = '../data/wine-equality-red.csv'

	if command == 'clean_and_output':
		# Removes rows with missing values and saves into a new csv
		remove_nan_and_save(raw_data_path, data_path)

	elif command == 'explore_features':
		# Explores the distribution of features
		explore_features(data_path, save_fig)

	elif command == 'tune':
		tune(data_path, save_fig)

	elif command == 'learning_rate':
		learn(data_path, save_fig)

	else:
		print("Invalid command")