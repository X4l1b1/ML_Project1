# ********************************************************** #
# Project 1 : CS-433 Machine Learning Class                  #
# Various regression models to detect Higgs particles        #
# Authors: Arthur Passuello, Francois Quellec, Julien Muster #
# ********************************************************** #

import numpy as np
from implementations import ridge_regression, predict_labels
from tools import load_csv_data, create_csv_submission, prepareData

# Data input and output paths
DATA_TRAIN_PATH = '../data/train.csv' 
DATA_TEST_PATH = '../data/test.csv'
OUTPUT_PATH = '../data/predictions_out.csv'


def main():
	# Model Parameters
	degree = 13
	whis = 2.5
	lambda_ = 0.0001

	# Load the training data
	print("Loading the training Datas...")
	y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

	# Clean and prepare our data
	print("Clean and prepare the training datas...")
	y_train, tX_train, ids_train = prepareData(y, tX, ids, degree, whis)

	# Train our models
	print("Train the models...")
	weights_0, loss_0 = ridge_regression(y_train[0], tX_train[0], lambda_)
	weights_1, loss_1 = ridge_regression(y_train[1], tX_train[1], lambda_)
	weights_2, loss_2 = ridge_regression(y_train[2], tX_train[2], lambda_)
	weights_3, loss_3 = ridge_regression(y_train[3], tX_train[3], lambda_)

	# Load the dataset to predict
	print("Loading the testing Datas...")
	y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

	# Prepare the data in the same way as the train dataset
	print("Clean and prepare the testing datas...")
	y_test, tX_test, ids_test = prepareData(y_test, tX_test, ids_test, degree, whis)

	# Predict each class 
	print("Predict the testing datas...")
	y_pred_0 = predict_labels(weights_0, tX_test[0])
	y_pred_1 = predict_labels(weights_1, tX_test[1])
	y_pred_2 = predict_labels(weights_2, tX_test[2])
	y_pred_3 = predict_labels(weights_3, tX_test[3])

	# Concatenate the results
	y_pred = np.concatenate([y_pred_0, y_pred_1, y_pred_2, y_pred_3])
	ids_test = np.concatenate([ids_test[0], ids_test[1], ids_test[2], ids_test[3]])

	# Write the results in a csv file
	print("Writing the results...")
	create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

	print("DONE!, your predictions are available in ", OUTPUT_PATH)


if __name__ == "__main__":
    # execute only if run as a script
    main()