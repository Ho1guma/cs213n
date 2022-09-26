from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #X = (500, 3073)=(N,D), X[i] = (3073), W = (3073,10)=(D,C)
    #y = (500) = (N)
    #dW = (3073,10)

    for i in range(num_train):
        scores = X[i].dot(W) #(3073)*(3073,10)= (N,C)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
              continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
              dW[:, j] += X[i].T
              dW[:,y[i]] += -X[i].T
              loss += margin
              
    dW /= num_train
    loss /= num_train

    dW += reg*2*W
    loss += reg * np.sum(W * W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W) #(N,C)
    num_train = X.shape[0]
    correct_class_scores = scores[np.arange(num_train), y].reshape(num_train,1)
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[np.arange(num_train), y]=0 #j=y[i] continue 부분
    loss = np.sum(margins)
    loss = loss/num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margins[margins>0]=1
    margin_count = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] = -margin_count
    dW = np.dot(X.T, margins)
    dW = dW/num_train
    dW += reg*2*W
    #https://mainpower4309.tistory.com/28 참고


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
