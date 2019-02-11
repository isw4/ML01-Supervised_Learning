import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


def wine_learning_rate(X, Y, score_fn, save=False):

	# KNN, SVM, and DT: learning rate across dataset sizes
	plt.figure()
	plt.title('Learning Rate of Different ML Algorithms')
	plt.ylabel('Accuracy')
	plt.xlabel('Training Data Fraction')

	learn_fracs = [0.10, 0.20, 0.3, 0.40, 0.50, 0.60, 0.7, 0.80, 0.9, 1]

	model = KNeighborsClassifier(n_neighbors=11)
	knn_score = avg_learning_score(model, score_fn, learn_fracs, X, Y)
	plt.plot(learn_fracs, knn_score)

	model = SVC(C=1, kernel='poly', degree=3, max_iter=10000)
	svm_score = avg_learning_score(model, score_fn, learn_fracs, X, Y)
	plt.plot(learn_fracs, svm_score)

	model = DecisionTreeClassifier(max_depth=22)
	dt_score = avg_learning_score(model, score_fn, learn_fracs, X, Y)
	plt.plot(learn_fracs, dt_score)

	plt.legend(['KNN', 'SVM', 'Decision Tree'])

	if save:
		plt.savefig('../graphs/wine/learning_data_fraction.png', dpi=300)

	# Boosting: learning rate across iterations
	n_shuffles = 10     # Number of times to run the model to average over the scores
	n_estimators = 80  # Hyperparameter of the model: number of classifiers/iterations
	boost_test_score = np.zeros((n_shuffles, n_estimators))
	boost_train_score = np.zeros((n_shuffles, n_estimators))
	for i in range(0, n_shuffles):
		model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=0.1)
		boost_train_score[i,:], boost_test_score[i,:] = boosting_iteration_score(model, score_fn, X, Y)
	boost_test_score = np.mean(boost_test_score, axis=0)
	boost_train_score = np.mean(boost_train_score, axis=0)

	plt.figure()
	plt.title('Accuracy Across Iterations of the Adaboost Classifier')
	plt.ylabel('Accuracy')
	plt.xlabel('Iteration')
	iteration_range = np.arange(1, n_estimators+1)
	plt.plot(iteration_range, boost_train_score)
	plt.plot(iteration_range, boost_test_score)
	plt.legend(['Training Score', 'Testing Score'])

	if save:
		plt.savefig('../graphs/wine/learning_boosting_iter.png', dpi=300)

	# Neural Nets: learning rate across iterations
	hidden_layer_sizes = (25, 25, 25, 25, 25)
	model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=2000)
	neuraln_train_score, neuraln_test_score = neural_net_iteration_score(model, score_fn, X, Y, epochs=200)

	plt.figure()
	plt.plot(neuraln_train_score, label='Train')
	plt.plot(neuraln_test_score, label='Test')
	plt.title("Accuracy Across Epochs for the Neural Net")
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.legend()

	if save:
		plt.savefig('../graphs/wine/learning_neuraln_iter.png', dpi=300)


def pet_learning_rate(X, Y, score_fn, save=False):
	# KNN, SVM, and DT: learning rate across dataset sizes
	plt.figure()
	plt.title('Learning Rate of Different ML Algorithms')
	plt.ylabel('Accuracy')
	plt.xlabel('Training Data Fraction')

	learn_fracs = [0.10, 0.20, 0.3, 0.40, 0.50, 0.60, 0.7, 0.80, 0.9, 1]

	model = KNeighborsClassifier(n_neighbors=43)
	knn_score = avg_learning_score(model, score_fn, learn_fracs, X, Y)
	plt.plot(learn_fracs, knn_score)

	model = SVC(C=0.1, kernel='poly', degree=3, max_iter=10000)
	svm_score = avg_learning_score(model, score_fn, learn_fracs, X, Y)
	plt.plot(learn_fracs, svm_score)

	model = DecisionTreeClassifier(max_depth=5)
	dt_score = avg_learning_score(model, score_fn, learn_fracs, X, Y)
	plt.plot(learn_fracs, dt_score)

	plt.legend(['KNN', 'SVM', 'Decision Tree'])

	if save:
		plt.savefig('../graphs/pet/learning_data_fraction.png', dpi=300)

	# Boosting: learning rate across iterations
	# n_shuffles = 10  # Number of times to run the model to average over the scores
	# n_estimators = 368  # Hyperparameter of the model: number of classifiers/iterations
	# boost_test_score = np.zeros((n_shuffles, n_estimators))
	# boost_train_score = np.zeros((n_shuffles, n_estimators))
	# for i in range(0, n_shuffles):
	# 	model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=0.1)
	# 	boost_train_score[i, :], boost_test_score[i, :] = boosting_iteration_score(model, score_fn, X, Y)
	# boost_test_score = np.mean(boost_test_score, axis=0)
	# boost_train_score = np.mean(boost_train_score, axis=0)
	#
	# plt.figure()
	# plt.title('Accuracy Across Iterations of the Adaboost Classifier')
	# plt.ylabel('Accuracy')
	# plt.xlabel('Iteration')
	# iteration_range = np.arange(1, n_estimators + 1)
	# plt.plot(iteration_range, boost_train_score)
	# plt.plot(iteration_range, boost_test_score)
	# plt.legend(['Training Score', 'Testing Score'])

	# if save:
	# 	plt.savefig('../graphs/pet/learning_boosting_iter.png', dpi=300)

	# Neural Nets: learning rate across iterations
	# hidden_layer_sizes = (100, 100, 100, 100)
	# model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=2000)
	# neuraln_train_score, neuraln_test_score = neural_net_iteration_score(model, score_fn, X, Y, epochs=200)
	#
	# plt.figure()
	# plt.plot(neuraln_train_score, label='Train')
	# plt.plot(neuraln_test_score, label='Test')
	# plt.title("Accuracy Across Epochs for the Neural Net")
	# plt.ylabel('Accuracy')
	# plt.xlabel('Epochs')
	# plt.legend()
	#
	# if save:
	# 	plt.savefig('../graphs/pet/learning_neuraln_iter.png', dpi=300)


def avg_learning_score(model, score_fn, learn_fracs, X, Y):
	"""
	Calculates the learning rate of the model over different sizes of training data.

	1) Partitions the data into training and validation sets
	2) Partitions the training data further into various sized fractions
		a) Trains the model using the fractional training data
		b) Predicts the validation set, and evaluates according to the given score fn
	3) Returns the average score

	:param model: sklearn model with parameters already set
	:param score_fn: sklearn.metric.?, An sklearn metric function used to evaluate the classifier
	:param learn_fracs: list, various fractions of training data to use to train the classifier
	:param X: df, Features
	:param Y: df, True classes
	:return: ndarray, The average accuracy using the various fractions of training data
	"""
	# Frac of the data(after validation removed) for training
	n_shuffles = 100
	valid_frac = 0.1  # Fraction of the dataset used as validation

	score = np.zeros((n_shuffles, len(learn_fracs)))
	for shuffle in range(0, n_shuffles):
		# Shuffling the data, then cutting out a validation set
		cutoff = int(valid_frac * len(Y))
		ix = np.arange(0, len(Y))
		np.random.shuffle(ix)
		valid_ix = ix[:cutoff]  # Shuffled indices of validation data

		valid_X = X.loc[valid_ix]
		valid_Y = Y.loc[valid_ix]

		# Fitting and calculating accuracy for the model with different training set sizes
		train_ix = ix[cutoff:]  # Shuffled indices of training data
		learning_cutoffs = (np.array(learn_fracs) * len(train_ix)).astype(int)  # Indices to cutoff diff sizes of data
		for co in range(0, len(learn_fracs)):
			learn_ix = train_ix[:learning_cutoffs[co]]
			learn_X = X.loc[learn_ix]
			learn_Y = Y.loc[learn_ix]

			model.fit(learn_X, learn_Y)
			pred_Y = model.predict(valid_X)
			score[shuffle, co] = score_fn(valid_Y, pred_Y)

	score = np.mean(score, axis=0)
	return score


def boosting_iteration_score(model, score_fn, X, Y):
	# Splitting into train and test sets
	test_frac = 0.1
	ix = np.arange(0, len(Y))
	np.random.shuffle(ix)
	cutoff = int(test_frac * len(Y))

	X_train = X.loc[ix[cutoff:]]
	Y_train = Y.loc[ix[cutoff:]]
	X_test = X.loc[ix[:cutoff]]
	Y_test = Y.loc[ix[:cutoff]]

	# Fitting the model and getting the predictions at each iteration
	model.fit(X_train, Y_train)
	Y_train_pred_iter = model.staged_predict(X_train)
	Y_test_pred_iter = model.staged_predict(X_test)

	train_score = []
	for Y_train_pred in Y_train_pred_iter:
		train_score.append(score_fn(Y_train, Y_train_pred))

	test_score = []
	for Y_test_pred in Y_test_pred_iter:
		test_score.append(score_fn(Y_test, Y_test_pred))

	return train_score, test_score


def neural_net_iteration_score(model, score_fn, X, Y, epochs=50):
	# Splitting into train and test sets
	test_frac = 0.1
	ix = np.arange(0, len(Y))
	np.random.shuffle(ix)
	cutoff = int(test_frac * len(Y))

	X_train = X.loc[ix[cutoff:]].reset_index().drop(columns='index')    # Need to reset index or partial_fit will error
	Y_train = Y.loc[ix[cutoff:]].values     # Need to get an ndarray or partial_fit will error
	X_test = X.loc[ix[:cutoff]]
	Y_test = Y.loc[ix[:cutoff]]

	# Code to get epoch accuracy from:
	# https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
	N_TRAIN_SAMPLES = X_train.shape[0]
	N_EPOCHS = epochs
	N_BATCH = 128
	N_CLASSES = np.unique(Y_train)

	scores_train = []
	scores_test = []

	# EPOCH
	epoch = 0
	while epoch < N_EPOCHS:
		# SHUFFLING
		random_perm = np.random.permutation(X_train.shape[0])
		mini_batch_index = 0
		while True:
			# MINI-BATCH
			indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
			model.partial_fit(X_train.loc[indices], Y_train[indices], classes=N_CLASSES)
			mini_batch_index += N_BATCH

			if mini_batch_index >= N_TRAIN_SAMPLES:
				break

		# SCORE TRAIN
		Y_train_pred = model.predict(X_train)
		scores_train.append(score_fn(Y_train, Y_train_pred))

		# SCORE TEST
		Y_test_pred = model.predict(X_test)
		scores_test.append(score_fn(Y_test, Y_test_pred))

		epoch += 1

	return scores_train, scores_test