import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
    scores = np.dot(X[i], W) # 1 * 10
    scores -= np.max(scores) 
    scores = np.exp(scores)
    correct_scores = scores[y[i]] 
  
    loss += -np.log(correct_scores / np.sum(scores))
    dW[:, y[i]] += -X[i]
    for j in range(num_class):
      dW[:, j] += scores[j] / np.sum(scores) * X[i]

  loss /= num_train
  dW /= num_train  
  
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train = X.shape[0]
  scores = np.dot(X, W) # 500 * 10
  t = np.max(scores, 1)
  scores -= np.reshape(t, (t.shape[0], -1))
  scores = np.exp(scores)
  correct_scores = scores[range(num_train), y]  # 500 * 1
  loss = np.mean(-np.log(correct_scores / np.sum(scores, 1)))

  loss += 0.5 * reg * np.sum(W * W)
  
  y_t = np.zeros_like(scores)
  y_t[range(num_train), y] = 1.0
  
  dW = -np.dot(X.T, y_t)
  m = np.sum(scores, 1)
  m = np.reshape(m, (m.shape[0], -1))
  dW += np.dot(X.T, scores / m)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

