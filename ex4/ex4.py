# ex4: Neural Network Basics

import scipy.io
import numpy as np
from scipy.optimize import minimize

import helperFunctions
import ex4modules

import os
if not(os.path.exists('./screenshots')):
    os.makedirs('./screenshots')

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

mat = scipy.io.loadmat('data1.mat')

X = mat["X"]
y = mat["y"]

m = X.shape[0]

# crucial step in getting good performance! changes the dimension from (m,1) to (m,)
# otherwise the minimization isn't very effective...
y=y.flatten() 

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100],:]
helperFunctions.displayData(sel)

import matplotlib.pyplot as plt
plt.savefig('./screenshots/ex4-1.png')
input('Program paused. Press enter to continue.\n')

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
mat = scipy.io.loadmat('weights.mat')
Theta1 = mat["Theta1"]
Theta2 = mat["Theta2"]

# Unroll parameters 
# ndarray.flatten() always creates copy (http://stackoverflow.com/a/28930580/583834)
# ndarray.ravel() requires transpose to have matlab unrolling order (http://stackoverflow.com/a/15988852/583834)
# np.append() always makes a copy (http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.append.html)
nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))

## ================ Part 3: Forward Propagation ================
#  To the neural network, you should first start by implementing neural network forward propagation.
#  You should complete the code in nnCostFunction() to return cost.
#  After implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('Performing Forward Propagation of Neural Network ...')

# Compute initial cost function value w/out regularization
J, _ = ex4modules.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0)

print('Initial cost at parameters loaded from ex4weights (w/out regularization): {:f} (this value should be about 0.287629)'.format(J))

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

# Compute initial cost function value w/ regularization
J, _ = ex4modules.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 1)

print('Initial cost at parameters loaded from ex4weights (w/ regularization): {:f} (this value should be about 0.383770)'.format(J))

## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function.

g = ex4modules.sigmoidGradient( np.array([1, -0.5, 0, 0.5, 1]) )
print('Sigmoid gradient evaluated at [1, -0.5, 0, 0.5, 1]: {}'.format(g))

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits.

print('Initializing Neural Network Parameters...')

initial_Theta1 = helperFunctions.randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = helperFunctions.randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate((initial_Theta1.reshape(initial_Theta1.size, order='F'), initial_Theta2.reshape(initial_Theta2.size, order='F')))

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network.
print('Checking Backpropagation... ')

#  Check gradients by running checkNNGradients
helperFunctions.checkNNGradients()

## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
print('Checking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
lambda_reg = 3
helperFunctions.checkNNGradients(lambda_reg)

## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('Training Neural Network...')

#  You can try different values of training_iterations or lambda_reg.
#  Note that scipy.optimize.minimize() can use a few different solver methods for gradient descent: 
#  http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
training_iterations = 20
lambda_reg = 0.1
myargs = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_reg)
results = minimize(ex4modules.nnCostFunction, x0=nn_params, args=myargs, options={'disp': True, 'maxiter': training_iterations}, method="L-BFGS-B", jac=True)

nn_params = results["x"]

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1), order='F')
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1), order='F')

## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('\nVisualizing Neural Network... \n')
helperFunctions.displayData(Theta1[:, 1:])

plt.savefig('./screenshots/ex4-2.png')

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = ex4modules.predict(Theta1, Theta2, X)

# uncomment code below to see the predictions that don't match
# fmt = '{}   {}'
# print(fmt.format('y', 'pred'))
# for y_elem, pred_elem in zip(y, pred):
#     if y_elem != pred_elem:
#         print(fmt.format(y_elem%10, pred_elem%10))

print('Training Set Accuracy: {:f}'.format( ( np.mean(pred == y)*100 ) ) )
input('Program finished. Press enter to exit.\n')


