import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_full_set(filepath):
	df = pd.read_csv(filepath)
	X = df.iloc[:, :-1]  # All columns but the last are features
	Y = df.iloc[:, -1]   # Last column is the class
	return X, Y


def get_separate_sets(filepath, test_frac=0.1):
	df = pd.read_csv(filepath)

	n_instances = df.shape[0]
	cutoff = int(test_frac * n_instances)   # index that partitions the dataset
	ix = np.arange(0, n_instances)          # indices of the dataset, to be shuffled and then partitioned
	np.random.shuffle(ix)

	# Partitioning the test and train set based on the shuffled ix
	test_X = df.iloc[ix[:cutoff], :-1]  # All columns but the last are features
	test_Y = df.iloc[ix[:cutoff], -1]   # Last column is the class

	train_X = df.iloc[ix[cutoff:], :-1]
	train_Y = df.iloc[ix[cutoff:], -1]

	return test_X, test_Y, train_X, train_Y


def normalize(X):
	# Mean normalization
	return (X - X.mean()) / X.std()


def plot_feature_distribution(X, Y, save_file_path=None):
	classes = Y.unique()

	# Dividing X by class
	X_class = {}
	for class_val in classes:
		X_class[class_val] = X[Y == class_val]

	print("====================================")
	print("Feature statistics:")
	for column in X.columns:
		feature = X.loc[:, column]

		# Dividing features by class
		feature_class = {}
		for class_val in classes:
			feature_class[class_val] = X.loc[Y == class_val, column]

		print("--------------------")
		print("Feature: {}".format(column))
		# Printing stats by class
		for class_val in classes:
			print("Y{} Mean:   {}".format(class_val, feature_class[class_val].mean()))
			print("Y{} Median: {}".format(class_val, feature_class[class_val].median()))

		# Plotting histogram by class
		plt.figure()
		plt.title("{} Distribution".format(column))
		if feature.dtype == 'bool':
			for class_val in classes:
				print(np.sum(feature_class[class_val] == False))
				print(np.sum(feature_class[class_val] == True))
				plt.bar([0, 1],
				        [np.sum(feature_class[class_val] == False),
				         np.sum(feature_class[class_val] == True)],
				        alpha=0.5)
			plt.xticks([0, 1], ('False', 'True'))
			plt.legend(["Class {}".format(x) for x in classes])
		elif feature.dtype == 'float':
			for class_val in classes:
				plt.hist(feature_class[class_val], 50, alpha=0.5)
			plt.legend(["Class {}".format(x) for x in classes])

		if save_file_path is not None:
			file_path = "{}/{}.png".format(save_file_path, column)
			plt.savefig(file_path, dpi=300)
	plt.show()


def plot_outcome_distribution(Y, save_file_path=None):
	print("====================================")
	print("Outcome statistics:")
	print("Min:    {}".format(Y.min()))
	print("Max:    {}".format(Y.max()))
	print("Mean:   {}".format(Y.mean()))
	print("Median: {}".format(Y.median()))
	print("Count:  {}".format(Y.size))
	plt.figure()
	plt.title("Outcome Distribution")
	plt.hist(Y, np.arange(Y.min(), Y.max() + 1), align='left')
	if save_file_path is not None:
		plt.savefig(save_file_path, dpi=300)
	plt.show()
