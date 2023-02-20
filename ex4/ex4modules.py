import numpy as np

# The expit() function in scipy.special module is the exact equivalent to the sigmoid function.
from scipy.special import expit as sigmoid

def sigmoidGradient(z):
    # This function computes the gradient of the sigmoid function evaluated at z.
    # If z is a vector or matrix, it returns the gradient for each element of z.

    # ====================== YOUR CODE HERE ======================

    # =============================================================
    return g

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg):
    # This function computes the cost and gradient of the neural network.

    # The parameters for the neural network are "unrolled" into the vector nn_params.
    # They need to be reshaped back into the weight matrices Theta1 and Theta2 for our 2 layer neural network.
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1), order='F')
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1), order='F')

    # Setup some useful variables
    m = len(X)
             
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros( Theta1.shape )
    Theta2_grad = np.zeros( Theta2.shape )

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the following parts.
    #
    # Part 1: Implement the forward propagation algorithm and return the cost in J.
    # Part 2: Implement the backpropagation algorithm. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and Theta2_grad, respectively.
    #         Note: The vector y passed into the function is a vector of labels containing values from 1..K.
    #               You need to map this vector into a binary vector of 1's and 0's to be used in cost function.
    # Part 3: Implement regularization with the cost function and gradients.
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    #

    # PART 1: FORWARD PROPAGATION

    # PART 2: BACK PROPAGATION

    # PART 3: REGULARIZATION FOR GRADIENT

    # =========================================================================

    # The returned parameter grad should be a "unrolled" vector of the partial derivatives of the neural network.
    grad = np.concatenate((Theta1_grad.reshape(Theta1_grad.size, order='F'), Theta2_grad.reshape(Theta2_grad.size, order='F')))

    return J, grad

def predict(Theta1, Theta2, X):
    # This function outputs the predicted label of X given the trained weights of a neural network (Theta1, Theta2)

    # turns 1D X array into 2D
    if X.ndim == 1:
        X = np.reshape(X, (-1,X.shape[0]))

    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros((m,1))

    # ====================== YOUR CODE HERE ======================

    # =========================================================================

    return p + 1 # offsets python's zero notation