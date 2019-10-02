# ***************************************************
# Project 1 : CS-433 Machine Learning Class
# Various regression models to detect Higgs particles
# Authors: Arthur Passuello, François Quellec
# ***************************************************
import numpy as np


def compute_mse_loss(y, tx, w):
    """Calculate the loss using mse"""
    e = y - tx@w
    return 1/(2*len(y)) * e@e.T

def compute_mae_loss(y, tx, w):
    """Calculate the loss using mae"""
    e = y - tx@w
    return 1/(2*len(y)) * np.sum(e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx@w
    return -1/len(y) * tx.T@e


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for n_iter in range(max_iters):
        # Compute loss and gradient then update accordingly the weights
        loss = compute_mae_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w = w- gamma * gradient

        # Print each iteration for debugging purpose
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    pass

def least_squares(y, tx):
    """Least squares regression using normal equations"""
    pass

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    pass

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD"""
    pass

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    pass

def custom_model(y, tx, initial_w, max_iters, gamma):
    """Custom"""
    pass