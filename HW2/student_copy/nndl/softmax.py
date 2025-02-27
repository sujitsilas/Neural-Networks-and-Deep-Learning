import numpy as np


class Softmax(object):

  def __init__(self, dims=[10, 3073]):
    self.init_weights(dims=dims)

  def init_weights(self, dims):
    """
    Initializes the weight matrix of the Softmax classifier.  
    Note that it has shape (C, D) where C is the number of 
    classes and D is the feature size.
    """
    self.W = np.random.normal(size=dims) * 0.0001

  def loss(self, X, y):
    """
    Calculates the softmax loss.
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    Inputs:
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
  
    Returns a tuple of:
    - loss as single float
    """

    loss = 0.0

    N = X.shape[0]

    scores = X.dot(self.W.T)

    exp_scores = np.exp(scores)  
    sums = np.sum(exp_scores, axis=1, keepdims=True)
    probs = exp_scores / sums 


    correct_log_probs = -np.log(probs[np.arange(N), y])
    loss = np.sum(correct_log_probs) / N

    return loss

  def loss_and_grad(self, X, y):
    """
    Same as self.loss(X, y), except that it also returns the gradient.

    Output: grad -- a matrix of the same dimensions as W containing 
      the gradient of the loss with respect to W.
    """

    loss = 0.0
    N = X.shape[0]

    grad = np.zeros_like(self.W)
    
    scores = X.dot(self.W.T)
    #scores -= np.max(scores, axis=1, keepdims=True)

    exp_scores = np.exp(scores)
    sums = np.sum(exp_scores, axis=1, keepdims=True)
    probs = exp_scores / sums

    correct_log_probs = -np.log(probs[np.arange(N), y])
    loss = np.sum(correct_log_probs) / N

    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N 

    # Now backprop into W
    # W is of shape (C, D), X is (N, D), dscores is (N, C)
    # => grad = dscores^T * X, shape = (C, D)
    grad = dscores.T.dot(X)

    return loss, grad


  def grad_check_sparse(self, X, y, your_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in these dimensions.
    """
  
    for i in np.arange(num_checks):
      ix = tuple([np.random.randint(m) for m in self.W.shape])
  
      oldval = self.W[ix]
      self.W[ix] = oldval + h # increment by h
      fxph = self.loss(X, y)
      self.W[ix] = oldval - h # decrement by h
      fxmh = self.loss(X,y) # evaluate f(x - h)
      self.W[ix] = oldval # reset
  
      grad_numerical = (fxph - fxmh) / (2 * h)
      grad_analytic = your_grad[ix]
      rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
      print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))

  def fast_loss_and_grad(self, X, y):
    """
    A vectorized implementation of loss_and_grad. It shares the same
    inputs and ouptuts as loss_and_grad.
    """
    loss = 0.0
    grad = np.zeros(self.W.shape)
  
    N = X.shape[0]

    # 1) Compute scores
    scores = X.dot(self.W.T)
    scores -= np.max(scores, axis=1, keepdims=True)

    # 2) Exponentiate and normalize
    exp_scores = np.exp(scores)
    sums = np.sum(exp_scores, axis=1, keepdims=True)
    probs = exp_scores / sums

    # 3) Compute loss
    correct_log_probs = -np.log(probs[np.arange(N), y])
    loss = np.sum(correct_log_probs) / N

    # 4) Compute gradient
    dscores = probs
    dscores[np.arange(N), y] -= 1
    dscores /= N
    grad = dscores.T.dot(X)

    return loss, grad


  def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1

    self.init_weights(dims=[np.max(y) + 1, X.shape[1]])

    loss_history = []

    for it in np.arange(num_iters):
      X_batch = None
      y_batch = None

      batch_indices = np.random.choice(num_train, batch_size)
      X_batch = X[batch_indices]
      y_batch = y[batch_indices]

      loss, grad = self.fast_loss_and_grad(X_batch, y_batch)
      loss_history.append(loss)

      self.W -= learning_rate * grad

      if verbose and it % 100 == 0:
          print('iteration %d / %d: loss %f' % (it, num_iters, loss))
      

    return loss_history

  def predict(self, X):
    """
    Inputs:
    - X: N x D array of training data. Each row is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[1])

    scores = X.dot(self.W.T)

    y_pred = np.argmax(scores, axis=1)

    return y_pred

