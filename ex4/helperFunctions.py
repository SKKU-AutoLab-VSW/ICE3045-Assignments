import numpy as np
import matplotlib.pyplot as plt

import myFunctions

def randInitializeWeights(L_in, L_out):
    # This function randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing connections. 
    # It is done to break the symmetry while training the neural network.
    # Note that W should be set to a matrix of size(L_out, 1 + L_in) as the first row of W corresponds to the parameters for the bias units

    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in)*(2*epsilon_init) - epsilon_init

    return W

def debugInitializeWeights(L_in, L_out):
    # This function pseudo-randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing connections. 
    # It is performed using sine function, which ensures that W is always of the same values. The returned weights are used for debugging purposes.
    # Note that W should be set to a matrix of size(L_out, 1 + L_in) as the first row of W corresponds to the parameters for the bias units

    # Set W to zeros
    W = np.zeros((L_out, 1 + L_in))

    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = np.reshape(np.sin(range(W.size)), W.shape) / 10

    return W

def computeNumericalGradient(J, theta):
    # This function computes the numerical gradient of the function J around theta.
    # It does so using "finite differences" and gives us a numerical estimate of the gradient.
    # Notes: The following code implements numerical gradient checking, and 
    #        returns the numerical gradient.It sets numgrad(i) to (a numerical 
    #        approximation of) the partial derivative of J with respect to the 
    #        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
    #        be the (approximately) the partial derivative of J with respect 
    #        to theta(i).)
    #                

    numgrad = np.zeros( theta.shape )
    perturb = np.zeros( theta.shape )
    e = 1e-4

    for p in range(theta.size):
        # Set perturbation vector
        perturb.reshape(perturb.size, order="F")[p] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        # Compute Numerical Gradient
        numgrad.reshape(numgrad.size, order="F")[p] = (loss2 - loss1) / (2*e)
        perturb.reshape(perturb.size, order="F")[p] = 0

    return numgrad

def checkNNGradients(lambda_reg=0):
    # This function creates a small neural network to check the backpropagation gradients.
    # It will output the analytical gradients produced by your backprop code and
    # the numerical gradients (computed using computeNumericalGradient).
    # These two gradient computations should result in very similar values.
    #

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(input_layer_size, hidden_layer_size)
    Theta2 = debugInitializeWeights(hidden_layer_size, num_labels)
    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(input_layer_size-1, m)
    y  = 1 + np.mod(range(m), num_labels).T

    # Unroll parameters
    nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))

    # Short hand for cost function
    def costFunc(p):
        return myFunctions.nnCostFunction(p, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, lambda_reg)

    _, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar. 
    fmt = '{:<25}{}'
    print(fmt.format('Numerical Gradient', 'Analytical Gradient'))
    for numerical, analytical in zip(numgrad, grad):
        print(fmt.format(numerical, analytical))

    print('If your backpropagation implementation is correct, the above two columns you get should be very similar.')

    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient(), then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

    print('Relative Difference: {:e} (this value should be less than 1e-9)'.format(diff))

def displayData(X, example_width=None):
    # This function displays 2D data stored in X in a nice grid.
    # It returns the figure handle h and the displayed array if requested.

	# closes previously opened figure. preventing a
	# warning after opening too many figures
	plt.close()

	# creates new figure 
	plt.figure()

    # turns 1D X array into 2D
	if X.ndim == 1:
		X = np.reshape(X, (-1,X.shape[0]))

	# Set example_width automatically if not passed in
	if not example_width or not 'example_width' in locals():
		example_width = int(round(np.sqrt(X.shape[1])))

	# Gray Image
	plt.set_cmap("gray")

	# Compute rows, cols
	m, n = X.shape
	example_height = int(round(n / example_width))

	# Compute number of items to display
	display_rows = int(np.floor(np.sqrt(m)))
	display_cols = int(np.ceil(m / display_rows))

	# Between images padding
	pad = 1

	# Setup blank display
	display_array = -np.ones((pad + display_rows * (example_height + pad),  pad + display_cols * (example_width + pad)))

	# Copy each example into a patch on the display array
	curr_ex = 1
	for j in range(1,display_rows+1):
		for i in range (1,display_cols+1):
			if curr_ex > m:
				break
		
			# Copy the patch
			
			# Get the max value of the patch to normalize all examples
			max_val = max(abs(X[curr_ex-1, :]))
			rows = pad + (j - 1) * (example_height + pad) + np.array(range(example_height))
			cols = pad + (i - 1) * (example_width  + pad) + np.array(range(example_width ))

			# Basic (vs. advanced) indexing/slicing is necessary so that we look can assign
			# 	values directly to display_array and not to a copy of its subarray.
			display_array[rows[0]:rows[-1]+1 , cols[0]:cols[-1]+1] = np.reshape(X[curr_ex-1, :], (example_height, example_width), order="F") / max_val
			curr_ex += 1
	
		if curr_ex > m:
			break

	# Display Image
	h = plt.imshow(display_array, vmin=-1, vmax=1)

	# Do not show axis
	plt.axis('off')

	plt.show(block=False)

	return h, display_array
