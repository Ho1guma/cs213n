from builtins import range
import numpy as np
from random import shuffle
from numpy.core.getlimits import warnings
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):    
      scores = X[i].dot(W)
      scores -= np.max(scores)
      loss += -scores[y[i]] + np.log(np.sum(np.exp(scores)))
      for j in range(num_classes):
        #정답 레이블(y[i])가 아닌 경우
        dW[:, j] += X[i]*(np.exp(scores[j])/np.sum(np.exp(scores)))
      #정답 레이블(y[i])인 경우    
      dW[:,y[i]] += -X[i]
    dW /= num_train
    loss /= num_train

    dW += reg*2*W
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores,axis=1).reshape(-1,1)
    loss = -np.sum(scores[np.arange(num_train), y]) + np.sum(np.log(np.sum(np.exp(scores),axis=1)))
    
    softmax = np.exp(scores)/np.sum(np.exp(scores), axis=1).reshape(-1,1)
    softmax[np.arange(num_train), y] -=1
    dW = X.T.dot(softmax)

    dW /= num_train
    loss /= num_train
    dW += reg*2*W
    loss += reg * np.sum(W * W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
