import numpy as np


def sample_zero_mean(x):
    """
    Zero-mean each sample
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    return x - np.mean(x, axis=1, keepdims=True)


def gcn(x, scale=55., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    std = np.sqrt(bias + np.mean(np.square(x), axis=1, keepdims=True))
    return scale * x / std


def feature_zero_mean(x, xtest):
    """
    Zero-mean along features. Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """
    mu = np.mean(x, axis=0, keepdims=True)
    return x - mu, xtest - mu


def zca(x, xtest, bias=0.1):
    """
    ZCA training data. Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features)) (assume mean=0)
    :param xtest: float32(shape=(samples, features))
    :param bias: bias to add to covariance matrix
    :return: tuple (x, xtest)
    """
    # Training data
    n = x.shape[0]
    sigma = (np.dot(x.T, x) / n)
    sigma += np.eye(sigma.shape[0]) * bias
    U, S, V = np.linalg.svd(sigma)
    pca = np.dot(np.dot(U, np.diag(1. / np.sqrt(S))), U.T)
    x = np.dot(x, pca)

    # Test data
    xtest = np.dot(xtest, pca)
    return x, xtest


def reshape_image(x, image_size):
    """
    Reshape and transpose image data.
    :param x: float32(shape=(samples, features))
    :return: 4-dimensional array float32(shape=(samples, channels, rows, cols))
    """
    return x.reshape((-1, 3, image_size, image_size))  # .transpose((0, 2, 3, 1))


def cifar_10_preprocess(x, xtest, image_size=32):
    """
    Load raw CIFAR10 data.
    sample_zero_mean and gcn xtrain and xtest.
    feature_zero_mean xtrain and xtest.
    zca xtrain and xtest.
    reshape_image xtrain and xtest.
    :param path: path of cifar10 data
    :return: tuple of tuples ((xtrain, ytrain), (xtest, ytest))
    """
    x, xtest = [sample_zero_mean(i) for i in (x, xtest)]
    x, xtest = [gcn(i) for i in (x, xtest)]
    x, xtest = feature_zero_mean(x, xtest)
    x, xtest = zca(x, xtest)
    x, xtest = [reshape_image(i, image_size) for i in (x, xtest)]
    return x, xtest
