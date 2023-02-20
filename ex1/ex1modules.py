import numpy as np

def computeCost(X, y, theta):
    # This function computes the cost of using theta as the
    # parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = len(y) # number of training examples

    # ====================== YOUR CODE HERE ===================================
    # TODO: Compute the cost of a particular choice of theimport numpy as np

def computeCost(X, y, theta):
    # This function computes the cost of using theta as the
    # parameter for linear regression to fit the data points in X and y

    # Initialize some useful values
    m = len(y); # number of training examples

    # ====================== YOUR CODE HERE ===================================
    # TODO: Compute the cost of a particular choice of theta
    #
    delta = np.dot(X,theta) - y; # h_theta(X) - y
    J = 1.0/(2*m) * sum(np.multiply(delta,delta))
    # =========================================================================
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    # This function performs gradient descent to learn theta.
    # It updates theta by taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = len(y); # number of training examples
    n = len(theta); # number of features per training example
    J_history = np.zeros((num_iters, 1))

    for iter in range(0,num_iters):

        # ====================== YOUR CODE HERE ======================
        # TODO: Perform a single gradient step on the parameter vector. 
        #
        delta = np.dot(X,theta) - y; # h_theta(X) - y
        theta_new = theta; # for init to same dimensions as theta

        for j in range(0,n):
            theta_new[j] = theta[j] - alpha * (1.0/m) * np.sum(np.multiply(delta, X[:,j]))
        theta = theta_new
        # ============================================================

        # Save the cost J in every iteration    
        J_history[iter] = computeCost(X, y, theta)

    return [theta, J_history]


def normalEqn(X, y):
    # This function computes the closed-form solution to linear regression
    # using the normal equations.

    # ====================== YOUR CODE HERE ======================
    # TODO: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.
    theta = np.linalg.inv(X.T*X)*X.T*y
    
    # ============================================================
    return theta
ta
    #

    # =========================================================================
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    # This function performs gradient descent to learn theta.
    # It updates theta by taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = len(y) # number of training examples
    n = len(theta) # number of features per training example
    J_history = np.zeros((num_iters, 1))

    for iter in range(0,num_iters):

        # ====================== YOUR CODE HERE ======================
        # TODO: Perform a single gradient step on the parameter vector. 
        #
        
        # ============================================================

        # Save the cost J in every iteration    
        J_history[iter] = computeCost(X, y, theta)

    return [theta, J_history]


def normalEqn(X, y):
    # This function computes the closed-form solution to linear regression
    # using the normal equations.

    # ====================== YOUR CODE HERE ======================
    # TODO: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.

    # ============================================================
    return theta
