import numpy as np

def initCentroids(X, K):
    # This function initializes K centroids that are to be used in K-Means on the dataset X
    # It sets the centroids to randomly chosen examples from the dataset X

    # You should return this values correctly
    centroids = np.zeros((K, X.shape[1]))

    # ====================== YOUR CODE HERE ======================


    # =============================================================

    return centroids

def findClosestCentroids(X, centroids):
    # This function computes the centroid memberships for every example
    # It returns the closest centroids in idx for dataset X where each row
    # is a single example. idx = m x 1 vector of centroid assignments
    # (i.e. each entry in range [1..K])
    #

    K = centroids.shape[0]

    # You need to return the following variables correctly.
    # Note that 8-bit integer data type is explicitly mentioned when
    # idx array is initialized. This is necessary since arrays used
    # as indices must be of integer (or boolean) type.
    idx = np.zeros((X.shape[0], 1), dtype=np.int8)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every example, find its closest centroid, and store
    #               the index inside idx at the appropriate location.
    #               Concretely, idx(i) should contain the index of the centroid
    #               closest to example i. Hence, it should be a value in the 
    #               range 1..K
    #
    # Note: You can use a for-loop over the examples to compute this.
    #

    # =============================================================
    return idx

def computeCentroids(X, idx, K):
    # This function returns the new centroids by computing the means of the data points assigned to each centroid.
    # It is given a dataset X where each row is a single data point,
    # a vector idx of centroid assignments (i.e. each entry in range [1..K]) for each example,
    # and K, the number of centroids.
    # You should return a matrix centroids, where each row is the mean of the data points assigned to it.
    #

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))


    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every centroid and compute mean of all points that
    #               belong to it. Concretely, the row vector centroids(i, :)
    #               should contain the mean of the data points assigned to
    #               centroid i.
    #
    # Note: You can use a for-loop over the centroids to compute this.
    #

    # =============================================================

    return centroids

