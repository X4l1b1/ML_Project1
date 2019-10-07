# ***************************************************
# Project 1 : CS-433 Machine Learning Class
# Various regression models to detect Higgs particles
# Authors: Arthur Passuello, François Quellec
# ***************************************************
import numpy as np
from proj1_helpers import *

def compute_mse_loss(y, tx, w):
    """Calculate the loss using mse"""
    error = y - tx@w
    loss = 1/(2*len(y)) * error@error
    return loss

def compute_mae_loss(y, tx, w):
    """Calculate the loss using mae"""
    error = y - tx@w
    loss = np.mean(np.abs(error))
    return loss

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    error = y - tx@w
    gradient = -1/len(y) * tx.T@error
    return gradient

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    return np.array([np.power(x,i) for i in range(degree + 1)]).T
    

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    loss = -1

    for n_iter in range(max_iters):
        # Compute loss and gradient then update accordingly the weights
        loss = compute_mae_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w -= gamma * gradient

        # Print each iteration for debugging purpose
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    loss = -1
    batch_size = 5
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        for n_iter in range(max_iters):
            # Compute loss and gradient then update accordingly the weights
            loss = compute_mae_loss(minibatch_y, minibatch_tx, w)
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            w -= gamma * gradient
            # Print each iteration for debugging purpose
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return loss, w

def least_squares(y, tx):
    """Least squares regression using normal equations"""
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    loss = compute_mse_loss(y, tx, w)
    return loss, w

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    omega = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    w = np.linalg.solve(tx.T@tx + omega, tx.T@y)
    loss = compute_mse_loss(y, tx, w)
    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    pass

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    pass

def custom_model(y, tx, initial_w, max_iters, gamma):
    """Custom"""
    pass