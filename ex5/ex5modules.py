import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def featureNormalize(X):
    # This function returns a normalized values of X where the mean
    # value of each feature is 0 and the standard deviation is 1.
    # This is often a good preprocessing step to do when
    # working with learning algorithms.

    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0)
    X_norm = X_norm/sigma

    return X_norm, mu, sigma

def learningCurve(X, y, Xval, yval, lambda_val):
    # This function generates the train and cross validation set errors needed 
    # to plot a learning curve. In particular, it returns two vectors of the
    # same length - error_train and error_val. Then, error_train(i) contains
    # the training error for i examples (and similarly for error_val(i)).
    # In this function, you will compute the train and test errors for
    # dataset sizes from 1 up to m. In practice, when working with larger
    # datasets, you might want to do this in larger intervals.
    #

    # Number of training examples
    m = len(X)

    # You need to return these values correctly
    error_train = np.zeros((m, 1))
    error_val   = np.zeros((m, 1))

    # ====================== YOUR CODE HERE ======================

    # =============================================================
                
    return error_train, error_val


def linRegCost(X, y, theta, lambda_val, return_grad=False):
    # This function computes the cost of using theta as the parameter
    # for linear regression to fit the data points in X and y. It returns
    # the cost in J and the gradient in grad.

    # Initialize some useful values
    m = len(y) # number of training examples

    # force to be 2D vector
    theta = np.reshape(theta, (-1,y.shape[1]))

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost and gradient of regularized linear 
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #

    # =============================================================

    if return_grad == True:
        return J, grad.flatten()
    elif return_grad == False:
        return J 

def plotFit(min_x, max_x, mu, sigma, theta, p):
    # This function plots the learned polynomial fit with power p 
    # and feature normalization (mu, sigma).

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.array(np.arange(min_x - 15, max_x + 25, 0.05)) # 1D vector

    # Map the X values 
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly/sigma

    # Add ones
    X_poly = np.column_stack((np.ones((x.shape[0],1)), X_poly))

    # Plot
    plt.plot(x, np.dot(X_poly, theta), '--', linewidth=2)

def polyFeatures(X, p):
    # This function takes a data matrix X (size m x 1) and
    #   maps each example into its polynomial features where
    #   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    #

    # ====================== YOUR CODE HERE ======================
    # Instructions: Given a vector X, return a matrix X_poly where the p-th 
    #               column of X contains the values of X to the p-th power.
    #

    # =========================================================================
    return X_poly


def trainLinearReg(X, y, lambda_val):
    # This function trains linear regression using the dataset (X, y)
    # and regularization parameter lambda_val. Returns the
    # trained parameters theta.
    #

    # Initialize Theta
    initial_theta = np.zeros((X.shape[1], 1))

    # Short hand for cost function to be minimized
    def costFunc(theta):
        return linRegCost(X, y, theta, lambda_val, True)

    # Now, costFunction is a function that takes in only one argument
    maxiter = 200
    results = minimize(costFunc, x0=initial_theta, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

    theta = results["x"]

    return theta


def validationCurve(X, y, Xval, yval):
    # This function generates the train and validation errors needed to
    # plot a validation curve that we can use to select lambda.
    # It returns the train and validation errors (in error_train, error_val)
    # for different values of lambda. You are given the training set (X, y)
    # and validation set (Xval, yval).
    #

    # Selected values of lambda (you should not change this)
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    # You need to return these variables correctly.
    error_train = np.zeros((len(lambda_vec), 1))
    error_val = np.zeros((len(lambda_vec), 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return training errors in 
    #               error_train and the validation errors in error_val. The 
    #               vector lambda_vec contains the different lambda parameters 
    #               to use for each calculation of the errors, i.e, 
    #               error_train(i), and error_val(i) should give 
    #               you the errors obtained after training with 
    #               lambda = lambda_vec(i)
    #

    # =========================================================================
    
    return lambda_vec, error_train, error_val