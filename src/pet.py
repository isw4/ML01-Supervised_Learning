import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import data_utils
import learning
from tuning import hyperparameter_tuning, test_and_print
from sklearn.metrics import accuracy_score


def drop_empty_rows(X, Y):
	row_no_nan = X.notna().all(axis=1)
	print("Number of rows with empty cells: {}".format(np.sum(~row_no_nan)))
	X = X[row_no_nan]
	Y = Y[row_no_nan]
	return X, Y


def aggregate_pet_labels(Y):
	Y_copy = Y.copy().str.strip().str.lower()
	def transform_labels(label):
		if label == 'adoption' or label == 'return_to_owner':
			return 0
		elif label == 'died' or label == 'euthanasia' or label == 'transfer':
			return 1
		return None
	Y_copy = Y_copy.apply(transform_labels)
	return Y_copy


def process_clean_and_save(filepath_input, filepath_output):
	### Read CSV and extract relevant features and the outcome
	df = pd.read_csv(filepath_input)
	X = df.loc[:, ['Name', 'AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color']]
	Y = df.loc[:, 'OutcomeType']

	### Process features
	# Encoding names into whether the pet has a name (pets with no name will be NaN)
	has_name = ~X.loc[:, 'Name'].isna()
	X['Named'] = has_name
	X = X.drop(columns='Name')

	# Dropping rows with missing features
	X, Y = drop_empty_rows(X, Y)

	# Encoding pet type(Dog or Cat) into whether the pet is a dog (only other type is cat)
	is_dog = X.loc[:, 'AnimalType'].str.strip().str.lower() == 'dog'
	X['isDog'] = is_dog
	X = X.drop(columns='AnimalType')

	# Transforming age from str to datetime. The types of ages that are available are given in 'x years',
	# 'x months', 'x weeks', 'x days'. We will transform this to days, following the assumption that
	# 1 year = 365.25 days
	# 1 month = 30.42 days
	# 1 week = 7 days
	raw_age = X.loc[:, 'AgeuponOutcome'].str.strip().str.lower().str.split(' ')
	def transform_age(pair):
		number = int(pair[0])
		time_unit = pair[1]
		if number == 0:
			return None
		elif time_unit == 'day' or time_unit == 'days':
			return int(number)
		elif time_unit == 'week' or time_unit == 'weeks':
			return int(number * 7)
		elif time_unit == 'month' or time_unit == 'months':
			return int(number * 30.42)
		elif time_unit == 'year' or time_unit == 'years':
			return int(number * 365.25)
		return None
	X['AgeuponOutcome'] = raw_age.apply(transform_age)

	# Dropping rows with age = NaN
	X, Y = drop_empty_rows(X, Y)

	# Processing Sex (values will be 'spayed/intact female' or 'spayed/intact male' or 'unknown')
	raw_sex = X.loc[:, 'SexuponOutcome'].str.strip().str.lower().str.split(' ')
	def transform_sex(pair):
		if len(pair) == 2:
			if pair[1] == 'male':
				return True
			if pair[1] == 'female':
				return False
		return None
	X['MaleuponOutcome'] = raw_sex.apply(transform_sex)
	X = X.drop(columns='SexuponOutcome')

	# Dropping rows with unknown sex
	X, Y = drop_empty_rows(X, Y)

	# Arranging the column ordering before adding the breed one-hot encoding
	X = X[['Named', 'isDog', 'AgeuponOutcome', 'MaleuponOutcome', 'Breed', 'Color']]

	# Processing breeds (one-hot encoding)
	# raw_breeds = X.loc[:, 'Breed'].str.strip().str.lower().str.split('/')
	# breeds = raw_breeds.apply(lambda x: x[0])
	# one_hot = pd.get_dummies(breeds)
	X = X.drop(columns='Breed')
	# X = X.join(one_hot)

	# Processing colour (one-hot encoding)
	# raw_color = X.loc[:, 'Color'].str.strip().str.lower().str.split('/')
	# color = raw_color.apply(lambda x: x[0])
	# one_hot = pd.get_dummies(color)
	X = X.drop(columns='Color')
	# X = X.join(one_hot)

	# Dropping empty rows just in case. All data is now processed
	drop_empty_rows(X, Y)

	print("Number of instances: {}".format(X.shape[0]))
	print("Number of features: {}".format(X.shape[1]))

	df = X.join(Y)
	df.to_csv(filepath_output, index=False)
	print('Output saved')


def explore_features(filepath, save_figures=False):
	X, Y = data_utils.get_full_set(filepath)
	Y = aggregate_pet_labels(Y)

	if save_figures:
		save_file_path = '../graphs/pet'
	else:
		save_file_path = None

	data_utils.plot_feature_distribution(X, Y, save_file_path)
	plt.show()


def tune(filepath, save_figures=False):
	### Getting and processing the dataset
	test_X, test_Y, train_X, train_Y = data_utils.get_separate_sets(filepath)

	# Normalize continuous features, and aggregate categories of pets
	train_X['AgeuponOutcome'] = data_utils.normalize(train_X['AgeuponOutcome'])
	train_Y = aggregate_pet_labels(train_Y)
	test_X['AgeuponOutcome'] = data_utils.normalize(test_X['AgeuponOutcome'])
	test_Y = aggregate_pet_labels(test_Y)


	### Running the tuning
	score_fn = accuracy_score
	score_fn_name = 'accuracy'

	print("Running KNN")
	if save_figures:
		save_path = '../graphs/pet/tuning_knn.png'
	else:
		save_path = None
	best_param, tuning_scores, model = hyperparameter_tuning('knn', score_fn_name, train_X, train_Y, save_path=save_path)
	test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, 'K-Nearest Neighbour', best_param, tuning_scores)

	print("Running SVM (linear)")
	if save_figures:
		save_path = '../graphs/pet/tuning_svm_linear.png'
	else:
		save_path = None
	best_param, tuning_scores, model = hyperparameter_tuning('svm_linear', score_fn_name, train_X, train_Y, save_path=save_path)
	test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, 'Support Vector Machines (linear)', best_param, tuning_scores)

	print("Running SVM (poly)")
	if save_figures:
		save_path = '../graphs/pet/tuning_svm_poly.png'
	else:
		save_path = None
	best_param, tuning_scores, model = hyperparameter_tuning('svm_poly', score_fn_name, train_X, train_Y, save_path=save_path)
	test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, 'Support Vector Machines (poly)', best_param, tuning_scores)

	print("Running decision trees")
	if save_figures:
		save_path = '../graphs/pet/tuning_dt.png'
	else:
		save_path = None
	best_param, tuning_scores, model = hyperparameter_tuning('dec_tree', score_fn_name, train_X, train_Y, save_path=save_path)
	test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, 'Decision Trees', best_param, tuning_scores)

	print("Running boosting")
	if save_figures:
		save_path = '../graphs/pet/tuning_boosting.png'
	else:
		save_path = None
	best_param, tuning_scores, model = hyperparameter_tuning('boosting', score_fn_name, train_X, train_Y, save_path=save_path)
	test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, 'Adaboost', best_param, tuning_scores)

	print("Running neural networks")
	if save_figures:
		save_path = '../graphs/pet/tuning_neuraln.png'
	else:
		save_path = None
	best_param, tuning_scores, model = hyperparameter_tuning('neural_n', score_fn_name, train_X, train_Y, save_path=save_path)
	test_and_print(model, score_fn, train_X, train_Y, test_X, test_Y, 'Neural Nets', best_param, tuning_scores)

	plt.show()


def learn(filepath, save_figures=False):
	X, Y = data_utils.get_full_set(filepath)
	X = data_utils.normalize(X)
	Y = data_utils.aggregate_wine_labels(Y)
	score_fn = accuracy_score

	learning.pet_learning_rate(X, Y, score_fn, save=save_figures)

	plt.show()


if __name__ == '__main__':
	command = sys.argv[1]

	if len(sys.argv) == 3 and sys.argv[2] == 'save':
		save_fig = True
	else:
		save_fig = False

	raw_data_path = '../data/raw-pet-outcomes.csv'
	data_path = '../data/pet-outcomes.csv'

	if command == 'clean_and_output':
		# Removes rows with missing values and saves into a new csv
		process_clean_and_save(raw_data_path, data_path)

	elif command == 'explore_features':
		# Explores the distribution of features
		explore_features(data_path, save_fig)

	elif command == 'tune':
		tune(data_path, save_fig)

	elif command == 'learning_rate':
		learn(data_path, save_fig)

	else:
		print("Invalid command")