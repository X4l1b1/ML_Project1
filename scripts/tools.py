# ********************************************************** #
# Project 1 : CS-433 Machine Learning Class                  #
# Various regression models to detect Higgs particles        #
# Authors: Arthur Passuello, Francois Quellec, Julien Muster #
# ********************************************************** #

import csv
import numpy as np
from implementations import *

#####################################################
#             Read/Write Functions                  #
#####################################################
def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


#####################################################
#              Features processing                  #
#####################################################
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.nanmean(x, axis=0)
    x = x - mean_x
    std_x = np.nanstd(x, axis=0)
    x = x / (std_x + 1)
    return x, mean_x, std_x

def standardize_with_mean_std(x, mean_x, std_x):
    """Standardize the original data set."""
    x = x - mean_x
    x = x / (std_x + 1)
    return x

def normalize(x):
    """Standardize the original data set."""
    max_x = np.max(x, axis=0)
    min_x = np.min(x, axis=0)
    x = (x-min_x) / (max_x-min_x)

    return x

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    ret = np.ones((len(x), 1))
    for i in range(1, degree + 1):
        ret = np.column_stack((ret, np.power(x,i)))
    return ret

def outliers_iqr(tX, whis = 2.5):   
    """ Cap the outliers to a lower and upper bound fixed at whis * 1st or 3nd quartile respectively."""
    for i, column in enumerate(tX.T):
        # Compute lower and upper bounds depending on the 1st and 3st quartile
        quartile_1, quartile_3 = np.nanpercentile(column, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * whis)
        upper_bound = quartile_3 + (iqr * whis)

        # Increase small outliers
        indices = np.where(column < lower_bound)
        tX[indices, i] = lower_bound

        # Decrease big outliers
        indices = np.where(column > upper_bound)
        tX[indices, i] = upper_bound
    return tX

def infereMissingValues(tX, col):
    """ Predict Nan values in tX with Ridge Regression """
    # Regression parameters
    lambda_ = 0.01
    degree = 3
    
    # Get all rows where DER_Mass_MMC is undefined and respectively defined
    train = tX[~np.isnan(tX[:, col]), :]
    test = tX[np.isnan(tX[:, col]), :]  
    
    # Create the training and testing sets
    tx_0_train = np.delete(train, col, axis=1)
    tx_0_train = tx_0_train[:, ~np.isnan(np.delete(tX, col, axis=1)).any(axis=0)]
    ty_0_train = train[:, col]
    tx_0_test =  np.delete(test, col, axis=1)
    tx_0_test = tx_0_test[:, ~np.isnan(np.delete(tX, col, axis=1)).any(axis=0)]
    
    # Expande the dimension with polynomial
    tx_0_train = build_poly(tx_0_train, degree)
    tx_0_test =build_poly(tx_0_test, degree)
    
    initial_w = np.zeros(tx_0_train.shape[1])
    # Train the model
    weights, loss = ridge_regression(ty_0_train, tx_0_train, lambda_)

    # Predict the undefined values
    tx_0_test = np.dot(tx_0_test, weights)
    tX[np.isnan(tX[:, col]), col]  = tx_0_test
 
    return tX

def pri_jet_split(y, tX, ids):
    """ Split the data into 4 differents class, depending on PRI_jet_num value """

    # Get the rows' indexes of each category
    indices_cat_0 = np.argwhere(tX[:, 22] == 0).flatten()
    indices_cat_1 = np.argwhere(tX[:, 22] == 1).flatten()
    indices_cat_2 = np.argwhere(tX[:, 22] == 2).flatten()
    indices_cat_3 = np.argwhere(tX[:, 22] == 3).flatten()

    # Split the dataset in 4
    tXC = [tX[indices_cat_0, :], tX[indices_cat_1, :], tX[indices_cat_2, :], tX[indices_cat_3, :]]
    idsC = [ids[indices_cat_0], ids[indices_cat_1], ids[indices_cat_2], ids[indices_cat_3]]
    yC = [y[indices_cat_0], y[indices_cat_1], y[indices_cat_2], y[indices_cat_3]]

    # Delete undefined features for each category, cf. features description : http://opendata.cern.ch/record/328
    # And the PRI_jet_num feature which is constant now
    tXC[0] = np.delete(tXC[0], (4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29), 1)
    tXC[1] = np.delete(tXC[1], (4, 5, 6, 12, 22, 26, 27, 28), 1)
    tXC[2] = np.delete(tXC[2], (22), 1)
    tXC[3] = np.delete(tXC[3], (22), 1)

    return yC, tXC, idsC

def prepareData(y, tX, ids, degree=None, whis=2.5):
    """Clean and do some feature engineering on the dataset."""
    # Replace -999 by nan
    tX[tX==-999]=np.nan

    # Cap our outliers records.
    tX = outliers_iqr(tX, whis)
      
    # Split the data into 4 differents class, depending on PRI_jet_num value
    yC, tXC, idsC = pri_jet_split(y, tX, ids)


    for i in range(len(tXC)):
        # Standardize the data
        tXC[i], _, _ = standardize(tXC[i])
        # Infere missing values on each column (normally just DER_Mass_MMC)     
        colsWithNan = np.unique(np.where(np.isnan(tXC[i]))[1])
        for col in colsWithNan:
            tXC[i] = infereMissingValues(tXC[i], col)
        
        # Build polynomial extension
        if degree is not None: 
            tXC[i] = build_poly(tXC[i], degree)
        
    return yC, tXC, idsC

#####################################################
#           Hyper-Parameters' Tuning                #
#####################################################

def splitData(y, tx, ratios=[0.4, 0.1]):
    """ Split the dataset into train, test and validation sets """ 
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    splits = (np.array(ratios) * len(y)).astype(int).cumsum()
    training_indices, validation_indices, test_indices = np.split(indices, splits)

    tX_train = tx[training_indices]
    y_train = y[training_indices]

    tX_validation = tx[validation_indices]
    y_validation = y[validation_indices]

    tX_test = tx[test_indices]
    y_test = y[test_indices]
    
    return tX_train, y_train, tX_validation, y_validation, tX_test, y_test

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def accuracy_score(true_y, preds):
    """ Compute the accuracy of predictions """
    return np.mean(np.array(preds) == np.array(true_y))

def cross_validate(y, x, k_fold, lambda_, degree, seed):
    """ Perform a k-fold validation """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data

    accuracies_tr = []
    accuracies_te = []
    for k in range(1, k_fold):            
        test_y = y[k_indices[k]]
        test_x = x[k_indices[k]]

        train_y = np.delete(y, k_indices[k])
        train_x = np.delete(x, k_indices[k], 0)
        
        if (degree >0):
            test_x = build_poly(test_x, degree)
            train_x = build_poly(train_x, degree)

        weight_tr, loss_tr = ridge_regression(train_y, train_x, lambda_)
        prediction_tr = predict_labels(weight_tr, train_x)
        prediction_te = predict_labels(weight_tr, test_x)

        accuracy_tr = accuracy_score(train_y, prediction_tr)
        accuracy_te = accuracy_score(test_y, prediction_te)

        accuracies_tr.append([accuracy_tr])
        accuracies_te.append([accuracy_te])
    
    accuracy_tr = np.mean(accuracies_tr)
    accuracy_te = np.mean(accuracies_te)
    return accuracy_te, accuracy_tr

