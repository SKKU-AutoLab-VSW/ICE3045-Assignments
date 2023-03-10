import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from ex5modules import *

#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

print('Loading and Visualizing Data ...')

mat = scipy.io.loadmat('ex5data1.mat')
X = mat["X"]
y = mat["y"]
Xval = mat["Xval"]
yval = mat["yval"]
Xtest = mat["Xtest"]
ytest = mat["ytest"]
m = X.shape[0]

# Plot training data
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show(block=False)

plt.ion()

#  We now compute cost and gradient.
#

print('Computing Cost and Gradient ...')

theta = np.array([[1] , [1]])
X_padded = np.column_stack((np.ones((m,1)), X))
J, grad = linRegCost(X_padded, y, theta, 1, True)

print('Cost at theta = [1  1]: {:f}\n(this value should be about 303.993192)\n'.format(J))
print('Gradient at theta = [1  1]:  [{:f} {:f}] \n(this value should be about [-15.303016 598.250744])'.format(grad[0], grad[1]))

#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 

#  Train linear regression with lambda = 0
theta = trainLinearReg(X_padded, y, 0)

# Plot training data and the fit over the data
plt.plot(X, np.dot(np.column_stack((np.ones((m,1)), X)), theta), '--', linewidth=2)
plt.show(block=False)

input('Program paused. Press enter to continue.\n')


#  Computing learning curve and plotting it is performed here.
#

error_train, error_val = learningCurve(np.column_stack((np.ones((m,1)), X)), y, np.column_stack((np.ones((Xval.shape[0], 1)), Xval)), yval, 0)

# resets plot 
plt.close()

p1, p2 = plt.plot(range(m), error_train, range(m), error_val)
plt.title('Learning curve for linear regression')
plt.legend((p1, p2), ('Train', 'Cross Validation'), numpoints=1, handlelength=0.5)
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.show(block=False)
plt.axis([0, 13, 0, 150])

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{:d}\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))

#  Note that the data is non-linear, so linear regression will not give a good fit.
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.column_stack((np.ones((m,1)), X_poly)) # Add Ones

# # Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.column_stack((np.ones((X_poly_test.shape[0],1)), X_poly_test)) # Add Ones

# # Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.column_stack((np.ones((X_poly_val.shape[0],1)), X_poly_val)) # Add Ones

print('Normalized Training Example 1:')
print('  {}  '.format(X_poly[1, :]))

input('Program paused. Press enter to continue.\n')

#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with 
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

lambda_val = 1
theta = trainLinearReg(X_poly, y, lambda_val)

# Plot training data and fit
# resets plot 
plt.close()
plt.figure(1)

plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plotFit(min(X), max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)') 
plt.ylabel('Water flowing out of the dam (y)')
plt.title ('Polynomial Regression Fit (lambda = {:f})'.format(lambda_val))
plt.show(block=False)

plt.figure(2)
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_val)
p1, p2 = plt.plot(range(1,m+1), error_train, range(1,m+1), error_val)

plt.title('Polynomial Regression Learning Curve (lambda = {:f})'.format(lambda_val))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 50])
plt.legend((p1, p2), ('Train', 'Cross Validation'))
plt.show(block=False)

print('Polynomial Regression (lambda = {:f})\n\n'.format(lambda_val))
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{:d}\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))

input('Program paused. Press enter to continue.\n')

#  You will now implement validationCurve to test various values of 
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

plt.close('all')
p1, p2 = plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.title('Polynomial Regression Validation Curve')
plt.legend((p1, p2), ('Train', 'Cross Validation'))
plt.xlabel('lambda')
plt.ylabel('Error')
plt.axis([0, 10, 0, 20])
plt.show(block=False)

print('lambda\t\tTrain Error\tValidation Error\n')
for i in range(len(lambda_vec)):
	print(' {}\t{}\t{}\n'.format(lambda_vec[i], error_train[i], error_val[i]))

input('Program paused. Press enter to continue.\n')


#  We now compute the test set error on the best lambda found.
#  Note that we're using X_poly - polynomial linear regression with polynomial features.
#  Because of this, we also have to use X_poly_test with polynomial features.
#

lambda_val = 3 # best lambda value from previous step
theta = trainLinearReg(X_poly, y, lambda_val)
error_test = linRegCost(X_poly_test, ytest, theta, 0)
print('Test set error: {:f}\n'.format(error_test)) # expected 3.859

input('Program paused. Press enter to continue.\n')


#  Learning curves can also be plotted with randomly selected examples.
#  We perform an example run in the following codes.
#

# lambda_val value for this step
lambda_val = 0.01

# number of iterations
times = 50

# initialize error matrices
error_train_rand = np.zeros((m, times))
error_val_rand   = np.zeros((m, times))

for i in range(1,m+1):
    for k in range(times):

        # choose i random training examples
        rand_sample_train = np.random.permutation(X_poly.shape[0])
        rand_sample_train = rand_sample_train[:i]

        # choose i random cross validation examples
        rand_sample_val   = np.random.permutation(X_poly_val.shape[0])
        rand_sample_val   = rand_sample_val[:i]

        # define training and cross validation sets for this loop
        X_poly_train_rand = X_poly[rand_sample_train,:]
        y_train_rand      = y[rand_sample_train]
        X_poly_val_rand   = X_poly_val[rand_sample_val,:]
        yval_rand         = yval[rand_sample_val]            

        # note that we're using X_poly_train_rand and y_train_rand in training
        theta = trainLinearReg(X_poly_train_rand, y_train_rand, lambda_val)
            
        # we use X_poly_train_rand, y_train_rand, X_poly_train_rand, X_poly_val_rand
        error_train_rand[i-1,k] = linRegCost(X_poly_train_rand, y_train_rand, theta, 0)
        error_val_rand[i-1,k]   = linRegCost(X_poly_val_rand,   yval_rand,    theta, 0)


error_train = np.mean(error_train_rand, axis=1)
error_val   = np.mean(error_val_rand, axis=1)

# resets plot 
plt.close()

p1, p2 = plt.plot(range(m), error_train, range(m), error_val)
plt.title('Polynomial Regression Learning Curve (lambda = {:f})'.format(lambda_val))
plt.legend((p1, p2), ('Train', 'Cross Validation'))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
plt.show(block=False)


print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{:d}\t\t{:f}\t{:f}\n'.format(i+1, error_train[i], error_val[i]))

input('Program paused. Press enter to continue.\n')
