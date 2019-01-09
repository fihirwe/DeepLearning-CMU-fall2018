import numpy as np
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
    x = np.array(x)
    y = np.array(y)

    return x.dot(y)


def Relu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if < 0 else x[i][j] for all pairs of indices i, j.

    """
    result = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                result[i][j] = 0
    return result

def vectorize_Relu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if < 0 else x[i][j] for all pairs of indices i, j.

    """
    # Write the vecotrized version here
    x = np.array(x)
    x[x<0]=0
    return x


def ReluPrime(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if x[i][j] < 0 else 1 for all pairs of indices i, j.

    """
    result = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                result[i][j] = 0
            else:
                result[i][j] = 1
    return result


def vectorize_PrimeRelu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if x[i][j] < 0 else 1 for all pairs of indices i, j.

    """
    # Write the vecotrized version here
    x = np.array(x)
    x[x<0]=0
    x[x>0]=1
    return x


def slice_fixed_point(x, l, start_point):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should have.
    start_point is an integer representing the point at which the final utterance should start in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    y=[]
    for e in x:
        y.append(e[start_point:start_point+l])
    return y


def slice_last_point(x, l):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    y=[]
    for e in x:
        y.append(e[-l:])
    return y


def slice_random_point(x, l):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    y=[]
    for e in x:
        e=np.array(e)
        start_point = np.random.randint(-l+e.shape[0]+1)
        #st.append([len(e),start_point])
        #if(start_point+l >len(e)-1):
         #   y.append(e[-l:])
        #else:
        y.append(e[start_point:start_point+l]) 
    y=np.array(y)
    return y


def pad_pattern_end(x):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.

    """
    x=np.array(x)
    lenh=[]
    for e in x:
        lenh.append(len(e))
    o=max(lenh)
    p=len(x[0][0])

    y=[]
    for e in x:
        y.append(np.pad(e, pad_width=(0, o-len(e)), mode='symmetric'))
    to_ret=[]
    for e in y:
        to_ret.append(slice_fixed_point(e,p,0))

    to_ret=np.array(to_ret)
    return to_ret


def pad_constant_central(x, c):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.

    """
    lenh=[]
    for e in x:
        lenh.append(len(e))
    o=max(lenh)
    p=len(x[0][0]) 

    to_ret=[]
    for e in x:
        if(o%2==0) & (len(e)%2==0):
            left=int((o-len(e))/2)
            right=int(((o-len(e))/2))
            k=np.pad(e, pad_width=(left,right), mode='constant', constant_values=c)
            to_ret.append(slice_fixed_point(k,p,left))
        elif((o%2!=0) & (len(e)%2!=0)):
            left=int((o-len(e))/2)
            right=int(((o-len(e))/2))
            k=np.pad(e, pad_width=(left,right), mode='constant', constant_values=c)
            to_ret.append(slice_fixed_point(k,p,left))
        else:
            left=int((o-len(e))/2)
            right=int(((o-len(e))/2)+1)
            k=np.pad(e, pad_width=(left,right), mode='constant', constant_values=c)
            to_ret.append(slice_fixed_point(k,p,left))
           
    to_ret=np.array(to_ret)
    return to_ret
