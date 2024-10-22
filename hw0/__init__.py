handout/hw0/._hw0.py                                                                                000644  000765  000024  00000000260 13332206057 015126  0                                                                                                    ustar 00sshaar                          staff                           000000  000000                                                                                                                                                                             Mac OS X            	   2   ~      �                                      ATTR       �   �                     �     com.apple.lastuseddate#PS    si[    ��C                                                                                                                                                                                                                                                                                                                                                    handout/hw0/hw0.py                                                                                  000644  000765  000024  00000010277 13332206057 014722  0                                                                                                    ustar 00sshaar                          staff                           000000  000000                                                                                                                                                                         import numpy as np
import os


def sumproducts(x, y):
    """ 
    x is a 1-dimensional int numpy array.
    y is a 1-dimensional int numpy array.
    Return the sum of x[i] * y[j] for all pairs of indices i, j.

    >>> sumproducts(np.arange(3000), np.arange(3000))
    20236502250000

    """
    result = 0
    for i in range(len(x)):
        for j in range(len(y)):
            result += x[i] * y[j]
    return result


def vectorize_sumproducts(x, y):
    """ 
    x is a 1-dimensional int numpy array. Shape of x is (N, ).
    y is a 1-dimensional int numpy array. Shape of y is (N, ).
    Return the sum of x[i] * y[j] for all pairs of indices i, j.

    >>> vectorize_sumproducts(np.arange(3000), np.arange(3000))
    20236502250000

    """
    # Write the vecotrized version here
    pass


def Relu(x):
    """ 
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if < 0 else x[i][j] for all pairs of indices i, j.

    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                x[i][j] = 0
    return x

def vectorize_Relu(x):
    """ 
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if < 0 else x[i][j] for all pairs of indices i, j.

    """
    # Write the vecotrized version here
    pass


def ReluPrime(x):
    """ 
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if x[i][j] < 0 else 1 for all pairs of indices i, j.

    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                x[i][j] = 0
            else:
                x[i][j] = 1
    return x
    

def vectorize_PrimeRelu(x):
    """ 
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if x[i][j] < 0 else 1 for all pairs of indices i, j.

    """
    # Write the vecotrized version here
    pass


def splice_fixed_point(x, l, start_point):
    """ 
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array. 
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utternaces that the final array should be in.
    start_point is an integer representing the point at which the final utterance should start in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)
    
    """
    pass


def splice_last_point(x, l):
    """ 
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array. 
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utternaces that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)
    
    """
    pass


def splice_random_point(x, l):
    """ 
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array. 
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utternaces that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1) 
    
    """
    pass


def pad_pattern_end(x):
    """ 
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array. 
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    
    Return a 3-dimensional int numpy array.
    
    """
    pass


def pad_constant_central(x, c):
    """ 
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array. 
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.
    
    """
    pass                                                                                                                                                                                                                                                                                                                                 handout/hw0/__init__.py                                                                             000644  000765  000024  00000000000 13327340263 015744  0                                                                                                    ustar 00sshaar                          staff                           000000  000000                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         