<header align ="right">Myles Johnson - 205868607 </header>
<h1 align="center"> ECE C147/C247: Neural Networks & Deep Learning, Winter 2023<br> Homework #2</br></h1>

## Noisy Linear Regression

### a) Express the expectation of the modified loss over the gaussian noise, in terms of the original loss plus a term independent of the dataset $\mathcal{D}$

$$
\mathbb{E}_{\delta\sim\mathcal{N}}[\~\mathcal{L}(\mathcal{\theta})] = \mathcal{L}(\mathcal{\theta}) + \mathcal{R}
\\
\~\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N(y^{(i)} - (x^{(i)}+ \delta^{(i)})^T\theta^2)
\\
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N(y^{(i)} - (x^{(i)})^T\theta^2)
$$

Simplify the inner term of the $\~\mathcal{L}(\theta)$ sum:

$= (y^{(i)} - (x^{(i)}+ \delta^{(i)})^T\theta^2)$

$= (y^{(i)} - (x^{(i)})^T\theta^2 - \delta^{(i)^T}\theta^2)$

$= (y^{(i)} - (x^{(i)})^T\theta)^2 - 2(y^{(i)}-(x^{(i)})^T\theta)((\delta^{(i)})^T\theta) + (\delta^{(i)^T}\theta^2)$

Since $\mathbb{E}$ is a linear operator, we can apply it to each term in the sum:

$\mathbb{E}_{\delta\sim\mathcal{N}}[(y^{(i)} - (x^{(i)}+ \delta^{(i)})^T\theta^2)] = \mathbb{E}_{\delta\sim\mathcal{N}}[(y^{(i)} - (x^{(i)})^T\theta)^2] - \mathbb{E}_{\delta\sim\mathcal{N}}[ 2(y^{(i)}-(x^{(i)})^T\theta)((\delta^{(i)})^T\theta)] + \mathbb{E}_{\delta\sim\mathcal{N}}[(\delta^{(i)^T}\theta^2)]$

Looking at the terms with $\delta$ in them, we can see that the first term is a constant, and the second term is a linear function of $\delta$.
So, we can apply the linearity of expectation to the second term:

$\mathbb{E}_{\delta\sim\mathcal{N}}[-2(y^{(i)}-(x^{(i)})^T\theta)((\delta^{(i)})^T\theta)]$

$= -2(y^{(i)}-(x^{(i)})^T\theta)\mathbb{E}_{\delta\sim\mathcal{N}}[(\delta^{(i)})^T\theta]$

Since $\mathbb{E}_{\delta\sim\mathcal{N}}[\delta^{(i)}] = 0 \in \mathbb{R}$:

$\mathbb{E}_{\delta\sim\mathcal{N}}[-2(y^{(i)}-(x^{(i)})^T\theta)((\delta^{(i)})^T\theta)] = 0$

The third term also contains $\delta$, we can apply the linearity of expectation to this term as well:

$\mathbb{E}_{\delta\sim\mathcal{N}}[(\delta^{(i)^T}\theta^2)]$

$= \mathbb{E}_{\delta\sim\mathcal{N}}[(\theta^T\delta^{(i)}\delta^{(i)^T}\theta)]$

$= \theta^T\mathbb{E}_{\delta\sim\mathcal{N}}[(\delta^{(i)}\delta^{(i)^T})]\theta$

Since $\mathbb{E}_{\delta\sim\mathcal{N}}[\delta^{(i)}\delta^{(i)^T}] = \sigma^2\bold{I}$:

$\mathbb{E}_{\delta\sim\mathcal{N}}[(\delta^{(i)^T}\theta^2)] = \sigma^2\theta^T\bold{I}\theta = \sigma^2 \Vert\theta\Vert_{2}^{2}$

Therefore, the overall expectation of modified loss is:

$\mathbb{E}_{\delta\sim\mathcal{N}}[(y^{(i)} - (x^{(i)}+ \delta^{(i)})^T\theta^2)]  = (y^{(i)} - (x^{(i)}+ \delta^{(i)})^T\theta^2) + \sigma^2 \Vert\theta\Vert_{2}^{2}$

where $\mathcal{L}(\theta) = (y^{(i)} - (x^{(i)}+ \delta^{(i)})^T\theta^2)$ so, $\mathcal{R} = \sigma^2 \Vert\theta\Vert_{2}^{2}$ which is not a function of $\mathcal{D}$.

### b) Based on your answer to (a), under expectation what regularization effect would the addition of the noise have on the model?

If R is equal to $\sigma^2 \Vert\theta\Vert_{2}^{2}$, the addition of noise to the model's parameters, as a regularization technique, would have the effect of adding a term to the loss function which is proportional to the L2-norm of the parameters, multiplied by $\sigma^2$.

### c) Suppose $\sigma \rightarrow 0$, what effect would this have on the model?

If $\sigma \rightarrow 0$, this term would become very small and have a negligible effect on the model. In this case, the model would not be regularized and could overfit to the training data.

### d) Suppose $\sigma \rightarrow \infty$, what effect would this have on the model?

On the other hand, if $\sigma \rightarrow \infty$, this term would become very large, and it would have a significant effect on the model. In this case, the model would be heavily regularized and could underfit to the training data. The model would be more robust to the noise but would be less accurate.

## 2. K-Nearest Neighbors

Code sections for this part are in `knn.py`:

```python
import numpy as np
import pdb


class KNN(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """
        Inputs:
        - X is a numpy array of size (num_examples, D)
        - y is a numpy array of size (num_examples, )
        """
        self.X_train = X
        self.y_train = y

    def compute_distances(self, X, norm=None):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.
        - norm: the function with which the norm is taken.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        if norm is None:
            norm = lambda x: np.sqrt(np.sum(x**2))
            # norm = 2

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in np.arange(num_test):

            for j in np.arange(num_train):
                # ================================================================ #
                # YOUR CODE HERE:
                #   Compute the distance between the ith test point and the jth
                #   training point using norm(), and store the result in dists[i, j].
                # ================================================================ #

                dists[i, j] = norm(X[i] - self.X_train[j])

                # ================================================================ #
                # END YOUR CODE HERE
                # ================================================================ #

        return dists

    def compute_L2_distances_vectorized(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train WITHOUT using any for loops.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # ================================================================ #
        # YOUR CODE HERE:
        #   Compute the L2 distance between the ith test point and the jth
        #   training point and store the result in dists[i, j].  You may
        #   NOT use a for loop (or list comprehension).  You may only use
        #   numpy operations.
        #
        #   HINT: use broadcasting.  If you have a shape (N,1) array and
        #   a shape (M,) array, adding them together produces a shape (N, M)
        #   array.
        # ================================================================ #

        dists = np.sqrt(
            ((X**2).sum(axis=1, keepdims=True))
            + (self.X_train**2).sum(axis=1)
            - 2 * X.dot(self.X_train.T)
        )

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance between the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in np.arange(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            # ================================================================ #
            # YOUR CODE HERE:
            #   Use the distances to calculate and then store the labels of
            #   the k-nearest neighbors to the ith test point.  The function
            #   numpy.argsort may be useful.
            #
            #   After doing this, find the most common label of the k-nearest
            #   neighbors.  Store the predicted label of the ith training example
            #   as y_pred[i].  Break ties by choosing the smaller label.
            # ================================================================ #

            closest_y = list(self.y_train[np.argsort(dists[i])[:k]])
            y_pred[i] = max(sorted(list(set(closest_y))), key=closest_y.count)

            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #

        return y_pred
```

(Attached workbook below)

## 3. Softmax Classifier Gradient

### Derive the log-likelihood $\mathcal{L}$, and its gradient w.r.t the parameters, $\nabla_{\bold{w_i}} \mathcal{L}$ and $\nabla_{b_i} \mathcal{L}$, for $i = 1,...,c$

We can group $\bold{w_i}$ and $b_i$ into a single vector by augmenting the data vectors with an additional dimension of constant 1. Let $\~x = \begin{bmatrix} \bold{x} \\ 1 \end{bmatrix}$, $\~w_i = \begin{bmatrix} \bold{w_i} \\ b_i \end{bmatrix}$, then $a_i(x) = \bold{w_i}^T \bold{x} + b_i = \~w_i^T \~x$. This unifies $\nabla_{\bold{w_i}} \mathcal{L}$ and $\nabla_{b_i} \mathcal{L}$ into a single gradient $\nabla_{\~w_i} \mathcal{L}$.

For a softmax classifier, the log-likelihood function is (via Discussion 3):

$\mathcal{L}(\bold{w_1},...,\bold{w_c},b_1,...,b_c) = \frac{1}{N}\sum_{n=1}^{N} \log \left( \frac{e^{(\bold{w_{y_n}}^T \bold{x_n} + b_{y_n})}}{\sum_{j=1}^{c}e^{(\bold{w_j}^T \bold{x_n} + b_j)}}\right)$

Using the NOTE that gives the single gradient $\nabla_{\~w_i} \mathcal{L}$ notation, the log-likelihood function becomes:

$\mathcal{L}(\~w_1,...,\~w_c) = \frac{1}{N}\sum_{n=1}^{N} \log \left( \frac{e^{(\~w_{y_n}^T ~x_n)}}{\sum_{j=1}^{c}e^{(\~w_j^T ~x_n)}}\right)$

And the gradient of the log-likelihood function with respect to $\~w_i$ becomes:

$\nabla_{\~w_i} \mathcal{L} = \frac{1}{N}\sum_{n=1}^{N} \left( \frac{e^{(\~w_i^T ~x_n)}}{\sum_{j=1}^{c}e^{(\~w_j^T ~x_n)}} - \mathbb{I}_{\{y_n = i\}}\right)~x_n$ $\space\space\space\space\space\because \mathbb{I}_{\{y_n = i\}}$ is an indicator function that is $1$ if $y_n = i$ and $0$ otherwise.

With this notation, we can express the gradient of the log-likelihood function with respect to a single vector $\~w_i$, which includes both the gradient with respect to the parameters of the $i$-th class, $\bold{w_i}$ and $b_i$, rather than expressing them separately. This gradient tells us how much the log-likelihood changes when we change the parameters $\~w_i$, and it takes into account all the data points. The first part of the equation, $\frac{e^{(~w_i^T ~x_n)}}{\sum_{j=1}^{c}e^{(~w_j^T ~x_n)}}$ , tells us the predicted probability of the $i$-th class for the $n$-th data point, and the second part, $\mathbb{I}_{\{y_n = i\}}$, tells us the true label for the $n$-th data point. By subtracting the true label from the predicted probability, we can see how well our model is doing for each data point, and by adding up the results for all data points, we can see how well our model is doing overall.

## 4. Hinge Loss Gradient

### Find the gradient of the loss function $\mathcal{L}(\bold{w},b)$ with respect to the parameters i.e $\nabla_{\bold{w}} \mathcal{L}$ and $\nabla_{b} \mathcal{L}$

$\mathcal{L}(\bold{w},b) = \frac{1}{K} \sum_{i=1}^{K}\text{hinge}_{y^{(i)}}(x^{(i)}) + \lambda\Vert\bold{w}\Vert$

Since the gradient is a linear operator, we can write the gradient of the loss function as the sum of the gradients of each term in the loss function:

$\nabla_{\bold{w}} \mathcal{L} = \frac{1}{K} \sum_{i=1}^{K}\nabla_{\bold{w}}\text{hinge}_{y^{(i)}}(x^{(i)}) + \lambda\nabla_{\bold{w}}\Vert\bold{w}\Vert$

Note that $\text{hinge}_{y^{(i)}}(x^{(i)}) = \max(0, 1 - y^{(i)}(\bold{w}^T x^{(i)} + b))$. So the gradient of the hinge loss is:

$\nabla_{\bold{w}}\text{hinge}_{y^{(i)}}(x^{(i)}) = \begin{cases} -y^{(i)}x^{(i)} & \text{if } 1 > y^{(i)}(\bold{w}^T x^{(i)} + b)  \\ 0 & \text{if } 1 < y^{(i)}(\bold{w}^T x^{(i)} + b) \end{cases}$

And the gradient of the norm is:

$\nabla_{\bold{w}}\Vert\bold{w}\Vert = \begin{cases} 1 & \text{if } \bold{w} > 0  \\ -1 & \text{if } \bold{w} < 0 \end{cases}$

So the gradient of the loss function is:

$\nabla_{\bold{w}} \mathcal{L} = \frac{1}{K} \sum_{i=1}^{K}\begin{cases} -y^{(i)}x^{(i)} & \text{if } 1 > y^{(i)}(\bold{w}^T x^{(i)} + b)  \\ 0 & \text{if } 1 < y^{(i)}(\bold{w}^T x^{(i)} + b) \end{cases} + \lambda\begin{cases} 1 & \text{if } \bold{w} > 0  \\ -1 & \text{if } \bold{w} < 0 \end{cases}$

## 5. Softmax Classifier

Code sections for this part are in `softmax.py`:

```python
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

        # Initialize the loss to zero.
        loss = 0.0

        # ================================================================ #
        # YOUR CODE HERE:
        #   Calculate the normalized softmax loss.  Store it as the variable loss.
        #   (That is, calculate the sum of the losses of all the training
        #   set margins, and then normalize the loss by the number of
        #   training examples.)
        # ================================================================ #

        # Keep track of the current training example
        i = 0

        # Iterates through each row of X multiplied by the transpose of the weight matrix
        for row in X.dot(self.W.T):

            # Subtract the max value of the row to prevent overflow when taking the exponential.
            row -= np.max(row)

            # Loss is calculated as -log(exp(row[y[i]]) / sum(exp(row))), where y[i] is the label for the current example
            loss += -np.log(np.exp(row[y[i]]) / sum(np.exp(row)))
            i = i + 1

        # Total loss is divided by the number of examples to get the average loss
        loss = loss / y.shape[0]

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss

    def loss_and_grad(self, X, y):
        """
        Same as self.loss(X, y), except that it also returns the gradient.

        Output: grad -- a matrix of the same dimensions as W containing
          the gradient of the loss with respect to W.
        """

        # Initialize the loss and gradient to zero.
        loss = 0.0
        grad = np.zeros_like(self.W)

        # ================================================================ #
        # YOUR CODE HERE:
        #   Calculate the softmax loss and the gradient. Store the gradient
        #   as the variable grad.
        # ================================================================ #

        # Calculate the dot product of W and X transpose
        activations = self.W.dot(X.T)

        # Calculate the element-wise exponential of a
        activations_exp = np.exp(activations)

        # Calculate the Score matrix
        score_matrix = activations_exp / np.sum(activations_exp, axis=0)

        # Subtract 1 from the corresponding element of Score where y=i
        np.subtract.at(score_matrix, (y, range(score_matrix.shape[1])), 1)

        # Calculate the gradient
        grad = np.dot(score_matrix, X)
        grad /= X.shape[0]

        # Calculate the loss
        loss = self.loss(X, y)

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss, grad

    def grad_check_sparse(self, X, y, your_grad, num_checks=10, h=1e-5):
        """
        sample a few random elements and only return numerical
        in these dimensions.
        """

        for i in np.arange(num_checks):
            ix = tuple([np.random.randint(m) for m in self.W.shape])

            oldval = self.W[ix]
            self.W[ix] = oldval + h  # increment by h
            fxph = self.loss(X, y)
            self.W[ix] = oldval - h  # decrement by h
            fxmh = self.loss(X, y)  # evaluate f(x - h)
            self.W[ix] = oldval  # reset

            grad_numerical = (fxph - fxmh) / (2 * h)
            grad_analytic = your_grad[ix]
            rel_error = abs(grad_numerical - grad_analytic) / (
                abs(grad_numerical) + abs(grad_analytic)
            )
            print(
                "numerical: %f analytic: %f, relative error: %e"
                % (grad_numerical, grad_analytic, rel_error)
            )

    def fast_loss_and_grad(self, X, y):
        """
        A vectorized implementation of loss_and_grad. It shares the same
        inputs and outputs as loss_and_grad.
        """
        loss = 0.0
        grad = np.zeros(self.W.shape)  # initialize the gradient as zero

        # ================================================================ #
        # YOUR CODE HERE:
        #   Calculate the softmax loss and gradient WITHOUT any for loops.
        # ================================================================ #

        # Compute the dot product of X and the transpose of W
        activations = X.dot(self.W.T)

        # Then subtract the maximum value of each row to prevent numerical overflow
        activations = (activations.T - np.amax(activations, axis=1)).T

        num_train = y.shape[0]

        # Compute the softmax scores for each sample
        activations_exp = np.exp(activations)
        score_matrix = np.zeros_like(activations_exp)
        score_matrix = activations_exp / np.sum(activations_exp, axis=1, keepdims=True)

        # Added small constant to prevent division by zero
        epsilon = 1e-7
        
        # Compute the loss
        loss = np.sum(
            -np.log(
                activations_exp[np.arange(activations.shape[0]), y]
                / (np.sum(activations_exp, axis=1) + epsilon)
            )
        )

        # Compute the gradient
        score_matrix[range(num_train), y] -= 1
        gradient_wrt_activations = score_matrix
        grad = gradient_wrt_activations.T.dot(X)
        grad /= num_train

        # Average the loss and gradient over the number of training samples
        loss = loss / num_train
        
        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss, grad

    def train(
        self, X, y, learning_rate=1e-3, num_iters=100, batch_size=200, verbose=False
    ):
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
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes

        self.init_weights(
            dims=[np.max(y) + 1, X.shape[1]]
        )  # initializes the weights of self.W

        # Run stochastic gradient descent to optimize W
        loss_history = []

        for it in np.arange(num_iters):
            X_batch = None
            y_batch = None

            # ================================================================ #
            # YOUR CODE HERE:
            #   Sample batch_size elements from the training data for use in
            #     gradient descent.  After sampling,
            #     - X_batch should have shape: (batch_size, dim)
            #     - y_batch should have shape: (batch_size,)
            #   The indices should be randomly generated to reduce correlations
            #   in the dataset.  Use np.random.choice.  It's okay to sample with
            #   replacement.
            # ================================================================ #

            # Randomly select a batch of training examples to update the weights with
            index = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[index]
            y_batch = y[index]

            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #

            # evaluate loss and gradient
            loss, grad = self.fast_loss_and_grad(X_batch, y_batch)
            loss_history.append(loss)

            # ================================================================ #
            # YOUR CODE HERE:
            #   Update the parameters, self.W, with a gradient step
            # ================================================================ #

            # Update the weights using the calculated gradient
            self.W = self.W - grad * learning_rate

            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #

            if verbose and it % 100 == 0:
                print("iteration {} / {}: loss {}".format(it, num_iters, loss))

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
        # ================================================================ #
        # YOUR CODE HERE:
        #   Predict the labels given the training data.
        # ================================================================ #

        # Compute the scores for each sample
        scores = X.dot(self.W.T)

        # Take the class with the highest score as the prediction
        y_pred = np.argmax(scores, axis=1)

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return y_pred
```

(Attached the softmax_nosol workbook below knn_nosol workbook)
