import numpy as np
import scipy.linalg as linalg

def pca(X):
    # This function runs principal component analysis on the dataset
    # represented as matrix X. It computes eigenvectors of the covariance matrix of X,
    # and returns the eigenvectors U, the eigenvalues (on diagonal) in S

    # Useful values
    m, n = X.shape

    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)

    # ====================== YOUR CODE HERE ======================




    # =============================================================

    return U, S


def projectData(X, U, K):
    # This function computes the projection of the normalized inputs X
    # into reduced dimensional space spanned by the first K columns of U,
    # which are the top K eigenvectors.

    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    # ====================== YOUR CODE HERE ======================


    # =============================================================

    return Z


def recoverData(Z, U, K):
    # This function recovers an approximation the original data
    # that has been reduced to K dimensions from projection.
    # It computes the approximation of the data by projecting back
    # onto the original space using the top K eigenvectors in U.

    # You need to return the following variables correctly.
    X_rec = np.zeros((Z.shape[0], U.shape[0]))

    # ====================== YOUR CODE HERE ======================


    # =============================================================

    return X_rec