# ***************************************************
# Project 1 : CS-433 Machine Learning Class
# Various regression models to detect Higgs particles
# Authors: Arthur Passuello, François Quellec
# ***************************************************
import numpy as np
from proj1_helpers import *


#####################################################
#             Loss Functions                        #
#####################################################
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

def compute_log_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1 + np.exp(tx@w)) - y*tx@w)


#####################################################
#              Activation Functions                 #
#####################################################
def sigmoid(t):
    """apply sigmoid activation function on t."""
    return 1/(1 + np.exp(-t))


#####################################################
#              Gradient Functions                   #
#####################################################
def calculate_least_square_gradient(y, tx, w):
    """Compute the gradient of the error for least square equation"""
    error = y - tx@w
    gradient = -1/len(y) * tx.T@error
    return gradient

def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss for logistic regression"""
    return tx.T @ (sigmoid(tx@w) - y)

#####################################################
#              Features processing                  #
#####################################################
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
    
#####################################################
#              Other Tools                          #
#####################################################
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Authors: Taken from the course homework 03
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


#####################################################
#              Models Definition                    #
#####################################################
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    loss = -1

    for n_iter in range(max_iters):
        # Compute loss and gradient then update accordingly the weights
        loss = compute_mae_loss(y, tx, w)
        gradient = calculate_least_square_gradient(y, tx, w)
        w -= gamma * gradient

        # Print each iteration for debugging purpose
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    # Final loss
    loss = compute_mae_loss(y, tx, w)

    # visualization
    print("loss={l}".format(l=loss))

    return loss, w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=5):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    loss = -1
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        for n_iter in range(max_iters):
            # Compute loss and gradient then update accordingly the weights
            loss = compute_mae_loss(minibatch_y, minibatch_tx, w)
            gradient = calculate_least_square_gradient(minibatch_y, minibatch_tx, w)
            w -= gamma * gradient
            # Print each iteration for debugging purpose
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    # Final loss
    loss = compute_mae_loss(y, tx, w)

    # visualization
    print("loss={l}".format(l=loss))

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
    """Logistic regression using gradient descent"""

     # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = compute_log_loss(y, tx, w)
        grad = compute_logistic_gradient(y, tx, w)
        w -= gamma*grad 
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    # Final loss
    loss = compute_log_loss(y, tx, w)

    # visualization
    print("loss={l}".format(l=loss))

    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD"""
    
     # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = compute_log_loss(y, tx, w) + lambda_*np.linalg.norm(w) 
        gradient = compute_logistic_gradient(y, tx, w) + 2*lambda_*w
        w -= gamma*gradient

        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    # Final loss 
    loss = calculate_loss(y, tx, w)

    # visualization
    print("loss={l}".format(l=loss))

    return loss, w
    
def custom_model(y, tx, initial_w, max_iters, gamma):
    """Custom"""
    pass