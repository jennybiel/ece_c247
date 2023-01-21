<h1 align="center"> ECE C147/C247: Neural Networks & Deep Learning, Winter 2023<br> Homework #1</br></h1>

## Linear Algebra Refresher

### a) Let $Q$ be a real orthogonal matrix

1. Q transpose and Q inverse are also orthonormal
   - Proof: $Q$ is orthonormal, which means that $Q^TQ = QQ^T = I$. Therefore, $Q^T=(Q^TQ)Q^T=IQ^T=Q^{-1}$. Additionally, since $QQ^T=I,Q^{-1}Q=I$ as well, which means $Q^{-1}$ is also orthonormal.

2. $Q$ has eigenvalues with norm 1
   - Proof: Since $Q$ is orthonormal, it preserves the Euclidean norm of vectors, meaning that the length of a vector is unchanged after being transformed by $Q$. Therefore, all of the eigenvalues of $Q$ must have a magnitude of 1, since otherwise the eigenvectors would change length.

3. The determinant of $Q$ is either $\pm1$
   - Proof: $Q$ is orthonormal, which means that $Q^TQ = QQ^T = I$. Therefore, the determinant of Q is the product of the eigenvalues, which all have magnitude 1, so the determinant must be either $\pm1$.

4. $Q$ defines a length preserving transformation.
   - Proof: Since $Q$ is orthonormal, it preserves the Euclidean norm of vectors. Therefore, it preserves the length of any vector, which means it defines a length-preserving transformation.

### b) Let A be a matrix

1. What is the relationship between the singular vectors of $A$ and the eigenvectors of $AA^T$ ? What about $A^TA$ ?

   - The relationship between the singular vectors of $A$ and the eigenvectors of $AA^T$ and $A^TA$ is that the singular vectors of $A$ are the eigenvectors of either $AA^T$ or $A^TA$ .

2. What is the relationship between the singular values of $A$ and the eigen-values of $AA^T$ ? What about $A^TA$ ?

   - The relationship between the singular values of $A$ and the eigenvalues of $AA^T$ and $A^TA$ is that the square of the singular values of $A$ are the eigenvalues of either $AA^T$ or $A^TA$ .

### c) True or False

1. Every linear operator in an $n$-dimensional vector space has n distinct eigenvalues

    - False. Not every linear operator in an $n$-dimensional vector space has $n$ distinct eigenvalues. An operator may have multiple eigenvectors corresponding to the same eigenvalue, in which case the number of distinct eigenvalues will be less than n.

2. A non-zero sum of two eigenvectors of a matrix $A$ is an eigenvector

   - False. A non-zero sum of two eigenvectors of a matrix $A$ is not necessarily an eigenvector. The sum of two eigenvectors of a matrix $A$ is an eigenvector if and only if the sum of the corresponding eigenvalues is 0.

3. If a matrix $A$ has the positive semidefinite property, i.e., $x^T Ax \ge 0$ for all $x$, then
its eigenvalues must be non-negative

   - True. If a matrix $A$ has the positive semidefinite property, i.e., $x^T Ax \ge 0$ for all $x$, then its eigenvalues must be non-negative. This is because the eigenvalues of a matrix are the roots of the characteristic polynomial, which is a polynomial of degree $n$. The roots of a polynomial are always real, and the roots of a polynomial of degree $n$ are always non-negative if the polynomial is positive semidefinite.
  
4. The rank of a matrix can exceed the number of distinct non-zero eigenvalues.

    - True. The rank of a matrix can exceed the number of distinct non-zero eigenvalues. This is because the rank of a matrix is the number of linearly independent columns, which may be less than the number of distinct non-zero eigenvalues.

5. A non-zero sum of two eigenvectors of a matrix $A$ corresponding to the same eigenvalue $\lambda$ is always an eigenvector.

   - False. A non-zero sum of two eigenvectors of a matrix $A$ corresponding to the same eigenvalue $\lambda$ is not always an eigenvector. The sum of two eigenvectors of a matrix $A$ corresponding to the same eigenvalue $\lambda$ is an eigenvector if and only if the sum of the corresponding eigenvalues is 0.

## Probability Refresher

### a) A jar of coins is equally populated with two types of coins. One is type “H50” and comes up heads with probability 0.5. Another is type “H60” and comes up heads with probability 0.6

1. You take one coin from the jar and flip it. It lands tails. What is the posterior probability that this is an $H50$ coin?

   - Use Bayes Theorem to calculate the posterior probability:
   - $P(H50|T)=\frac{P(T|H50)*P(H50)}{P(T)}=\frac{0.5*0.5}{(0.5*0.5+0.4*0.5)}=0.56$

2. You put the coin back, take another, and flip it 4 times. It lands $T,H,H,H$ . How likely is the coin to be type $H50$?

   - The likelihood of getting $T, H, H, H$ in 4 flips given that the coin is type $H50:P(T,H,H,H|H50)=(0.5)^3*(0.5)^1=0.0625$.
   - Use Bayes Theorem to calculate the posterior probability:
   - $P(H50|T,H,H,H)=\frac{P(T,H,H,H|H50)*P(H50)}{P(T,H,H,H)}=\frac{(0.5)^3*(0.5)^1*0.5}{P(T,H,H,H)}$
     - $P(T,H,H,H)=P(T|H50)*P(H,H,H|T,H50)*P(H50)+P(T|H60)*P(H,H,H|T,H60)*P(H60)=(1-0.5)*(0.5)^3+(1-0.6)*(0.6)^3=0.1489$
   - $P(H50|T,H,H,H)=\frac{(0.5)^3*(0.5)^1*0.5}{0.1489}\approx0.21$

3. A new jar is now equally populated with coins of type $H50,H55$, and $H60$ (with probabilities of coming up heads $0.5,0.55$, and $0.6$ respectively. You take one coin and flip it 10 times. It lands heads 9 times. How likely is the coin to be of each possible type?

   - Each coin is equally likely to be chosen, so the prior probability of each coin is $\frac{1}{3}$.
   - Use Bayes Theorem and Binomial Theorem, $\frac{n!}{k!(n-k!)}$ , to calculate the posterior probability for each coin flipping $9H\ in\ 10F$:

     - $P(H50|9H\ in\ 10F) = \frac{P(9H\ in\ 10F|H50) * P(H50)}{P(9H\ in\ 10F)}$
       - $P(H50|9H\ in\ 10F)=\frac{{10\choose9}*(0.5)^9*(0.5)^1*\frac{1}{3}}{P(9H\ in\ 10F)}$

     - $P(H55|9H\ in\ 10F) = \frac{P(9H\ in\ 10F|H55) * P(H55)}{P(9H\ in\ 10F)}$
       - $P(H55|9H\ in\ 10F)=\frac{{10\choose9}*(0.55)^9*(0.45)^1*\frac{1}{3}}{P(9H\ in\ 10F)}$

     - $P(H60|9H\ in\ 10F) = \frac{P(9H\ in\ 10F|H60) * P(H60)}{ P(9H\ in\ 10F)}$
       - $P(H60|9H\ in\ 10F)=\frac{{10\choose9}*(0.6)^9*(0.4)^1*\frac{1}{3}}{P(9H\ in\ 10F)}$

     - $P(9H\ in\ 10F)={10\choose9}(0.5)^9*(0.5)^1+{10\choose9}(0.55)^9*(0.45)^1+{10\choose9}(0.6)^9*(0.4)^1$

     - $P(H50|9H\ in\ 10F)\approx0.046$
     - $P(H55|9H\ in\ 10F)\approx0.098$
     - $P(H60|9H\ in\ 10F)\approx0.19$

### b) Students at UCLA are from these disciplines: 15% Science, 21% Healthcare, 24% Liberal Arts, and 40% Engineering. (Each student belongs to a unique discipline.) The students attend a lecture and give feedback. Suppose 90% of the Science students liked the lecture, 18% of the Healthcare students liked it, none of the Liberal Arts students liked it, and 10% of the Engineering students liked it. If a student is randomly chosen, and the student liked the lecture, what is the conditional probability that the student is from Science?

- Make $S$ the event that the student is from Science, $L$ the event that the student liked the lecture, then P(S|L) is the conditional probability that the student is from Science given that the student liked the lecture.
- $P(S)=0.15$
  - The probability that a student is from Science
- $P(L|S)=0.9$
  - The probability that a student liked the lecture given that the student is from Science

- Also make H, A, E are the events that the student is from Healthcare, Liberal Arts and Engineering respectively
  
- $P(H)=0.21$
- $P(B|H)=0.18$
- $P(A)=0.24$
- $P(B|A)=0$
- $P(E)=0.4$
- $P(B|E)=0.1$

- The total Probability to calculate the probability that a student liked the lecture:
  - $P(L)=P(L|S)P(S)+P(L|H)P(H)+P(L|A)P(A)+P(L|E)P(E)=0.9*0.15+0.18*0.21+0*0.24+0.1*0.4=0.153$

- So, the conditional probability that the student is from Science given that the student liked the lecture is:

  - $\frac{P(S|L)=P(L|S)*P(S)}{P(L)}=\frac{0.9*0.15}{0.153}=0.59$

### c) Consider a pregnancy test with the following statistics

#### "If the woman is pregnant, the test returns “positive” (or 1, indicating the woman is pregnant) 99% of the time. If the woman is not pregnant, the test returns “positive” 10% of the time. At any given point in time, 99% of the female population is not pregnant."

#### What is the probability that a woman is pregnant given she received a positive test? The answer should make intuitive sense; give an explanation of the result that you find

- Given that a woman received a positive test, the probability that she is pregnant is:

- $P(Preg|T)=\frac{P(T|Preg)*P(Preg)}{P(T)}$
  - $Preg$ is the event that the woman is pregnant
  - $T$ is the event that the test returns positive

- $P(Preg)$ is the probability that a woman is pregnant ($1\%$ or $0.01$)
- $P(T|preg)$ is the probability that the test returns positive given that the woman is pregnant ($99\%$ or $0.99$)
- $P(T)$ is the overall probability of a positive test result and can be calculated as:
  - $P(T)=P(T|A)*P(A)+P(T|Preg')*P(Preg')$

    - $P(T|Preg')$ is the probability that the test returns positive given that the woman is not pregnant ($10\%$ or $0.1$)
    - $P(Preg')$ is the event that the woman is not pregnant ($99\%$ or $0.99$)
  - Thus, $P(T) = 0.99*0.01 + 0.1*0.99 = 0.108$

- Therefore $P(Preg|T)=\frac{0.99*0.01}{0.108}=0.0917$ or $9.17\%$

- This makes sense since a positive test result doesn't necessarily mean that the woman is pregnant, even though the test returns positive 99% of the time when the woman is pregnant. The test also returns positive 10% of the time when the woman is not pregnant. So, that combined with the fact the majority of the female population is not pregnant, a positive test result by itself is not a strong indicator of a pregnancy.

### d) Let $x_1,x_2,...,x_n$ be identically distributed random variables. A random vector, $x$, is defined as:<h3 align="center">$x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$</h3><h3> What is $\mathbb{E}(Ax + b)$ in terms of $\mathbb{E}(x)$, given that $A$ and $b$ are deterministic?</h3>

Expected value is the weighted average of possible values of a random variable, with weights given by their respective theoretical probabilities. The formula for expected value is:

$$
\mathbb{E}(x) = \sum_{i=1}^n x_i P(x_i)
$$

First since $A$ and $b$ are deterministic, they can be pulled out of the Expected value equation:

$\mathbb{E}(Ax + b) = A\mathbb{E}(x) + b$

Additionally, since $x_1,x_2,\dots,x_n$ are identically distributed, they have the same expected value $\mathbb{E}(x)$, and since A is a deterministic matrix, it doesn't change the expected value of the vector.

### e) Let <h3 align="center">$\mathrm{cov}(x) = \mathbb{E}((x-\mathbb{E}x)(x-\mathbb{E}x)^T)$</h3><h3>What is $\mathrm{cov}(Ax + b)$ in terms of $\mathrm{cov}(x)$, given that $A$ and $b$ are deterministic?</h3>

Covariance is a measure of how two random variables change together. The formula for covariance is:

$$
\mathrm{cov}(X) = \mathbb{E}[(X - \mathbb{E}(X))(X - \mathbb{E}(X))^T]
$$

So given $\mathrm{cov}(Ax + b)$ in terms of $\mathrm{cov}(x)$, we can write:

$\mathrm{cov}(Ax + b) = \mathbb{E}[((Ax + b) - \mathbb{E}(Ax + b))((Ax + b)- \mathbb{E}(Ax + b))^T]$

$A$ and $b$ are deterministic, so they can be pulled out of the Expected values:

$\mathrm{cov}(Ax + b) = \mathbb{E}[A(x - \mathbb{E}(x)) + (b - \mathbb{E}(b))(A(x - \mathbb{E}(x)) + (b - \mathbb{E}(b)))^T]$

We can simplify the above expression using the properties of the Expected value:

$\mathbb{E}x = \mathbb{E}(Ax + b) = A\mathbb{E}x + b$

After substituting the above expression, we get:

$\mathrm{cov}(Ax + b) = A\mathbb{E}((x - \mathbb{E}x)(x - \mathbb{E}x)^T)A^T$

$\space\space\space\space\space= A\mathrm{cov}(x)A^T$

So $\mathrm{cov}(Ax + b) = A\mathrm{cov}(x)A^T$ maening the covariance matrix of $Ax+b$ is equal to the covariance matrix of $x$

## Multivariate Derivatives

### a) Let $x \in \R^{n}, y \in \R^{m},$ and $A \in \R^{n \times m}$. What is $\nabla_{x} x^TAy$?

$x^TAy = \sum_{i=1}^{n}\sum_{j=1}^{m}a_{ij}x_{i}y_{j}$

$\nabla_{x} x^TAy = \frac{\partial}{\partial x_i}(\sum_{i=1}^{n}\sum_{j=1}^{m}a_{ij}x_{i}y_{j}) = \sum_{i=1}^{n}\sum_{j=1}^{m}a_{ij}y_{j} = A^Ty$

### b)Let $x \in \R^{n}, y \in \R^{m},$ and $A \in \R^{n \times m}$. What is $\nabla_{y} x^TAy$?

$\nabla_{y} x^TAy = \frac{\partial}{\partial y_j}(\sum_{i=1}^{n}\sum_{j=1}^{m}a_{ij}x_{i}y_{j})= \sum_{i=1}^{n}\sum_{j=1}^{m}a_{ij}x_{i} = x^TA$

### c)Let $x \in \R^{n}, y \in \R^{m},$ and $A \in \R^{n \times m}$. What is $\nabla_{A} x^TAy$?

$\nabla_{A} x^TAy = \frac{\partial}{\partial A_{ij}}(\sum_{i=1}^{n}\sum_{j=1}^{m}a_{ij}x_{i}y_{j}) = \sum_{i=1}^{n}\sum_{j=1}^{m}x_{i}y_{j} = x^Ty$

### d) Let $x \in \R^{n}, A \in \R^{n \times n},$ and let $f(x) = x^TAx+b^{T}x$. What is $\nabla_{x} f$?

$\frac{\partial}{\partial x_i}(x^TAx+b^{T}x) = \frac{\partial}{\partial x_i}(x^TAx) + \frac{\partial}{\partial x_i}(b^{T}x)$

$\frac{\partial}{\partial x_i}(x^TAx) = \frac{\partial}{\partial x_i}(\sum_{i=1}^{n}\sum_{j=1}^{n} a_{ij}x_{i}x_{j}) = 2Ax \space\space\space\space\space \because \text{A is symetric}$

$\frac{\partial}{\partial x_i}(b^{T}x) = \frac{\partial}{\partial x_i}(\sum_{j=1}^{n} b_jx_j) = b$

Therefore,

$\nabla_{x} f = 2Ax+b$

### e) Let $A \in \R^{n \times n}$ and $B \in \R^{n \times n}$. and $f=tr(AB)$. What is $\nabla_{A} f$?

$\mathcal{tr}(AB) = \sum_{i=1}^{n}a_{ii}b_{ii}$

$\nabla_{A} f = \frac{\partial}{\partial A_{ii}}(\sum_{i=1}^{n}a_{ii}b_{ii}) = \sum_{i=1}^{n}b_{ii} = B$

### f) Let $A \in \R^{n \times n}$ and $B \in \R^{n \times n}$. and $f=tr(BA+A^TB+A^2B)$. What is $\nabla_{A} f$?

Since the derivative of trace of a matrix is the transpose of the matrix, we can breakdown the expression as:

$\frac{\partial}{\partial A_{i,j}} \mathcal{tr}(BA) = B^T$

$\frac{\partial}{\partial A_{i,j}} \mathcal{tr}(A^TB) = B$

$\frac{\partial}{\partial A_{i,j}} \mathcal{tr}(A^2B) = 2AB$

So the gradient of this function with respect to A is the sum of these three derivatives is $B^T + B + 2AB$ or
$\nabla_{A} f = B^T + B + 2AB$.

### g) Let $A \in \R^{n \times n}$ and $B \in \R^{n \times n}$. and $f=\Vert A+\lambda B \Vert_{F}^{2}$. What is $\nabla_{A} f$?

$\nabla_{A} f = 2(A+\lambda B)$

The Frobenius norm of a matrix is defined as the square root of the sum of the squares of its elements:

$$
\Vert C \Vert_{F}^{2} = \sum_{i=1}^{n} \sum_{j=1}^{n} C_{i,j}^2
$$

Using the chain rule, we can derive the derivative of the Frobenius norm with respect to a matrix:

$\frac{\partial}{\partial A_{i,j}} \Vert A+\lambda B \Vert_{F}^{2} = \frac{\partial}{\partial A_{ij}} (\sum_{i=1}^{n} \sum_{j=1}^{n} (A_{ij}+\lambda B_{ij})^2)$

$\space\space\space\space\space= 2(A_{ij}+\lambda B_{ij})$

Therefore,

$\nabla_{A} f = 2(A+\lambda B)$

## Deriving Least-Squares With Matrix Derivatives

In least-squares, we seek to estimate some multivariate output $y$ via the model

$$
\hat{y}=Wx
$$

In the training set we’re given paired data examples $(x^{(i)},y^{(i)})$ from $i = 1,...,n$ . Least-squares is the following quadratic optimization problem:

$$
\min_{W}\frac{1}{2}\sum_{i=1}^{n}\Vert y^{(i)}-Wx^{(i)}\Vert^2
$$

Derive the optimal $W$

Where $W$ is a matrix, and for each example in the training set, both $x^{(i)}$ and $y^{(i)}\ \forall \ i = 1,...,n$ are vectors

**SOLUTION:**

$\frac{1}{2}\sum_{i=1}^{n}\Vert y^{(i)}-Wx^{(i)}\Vert^2 = \frac{1}{2}\mathcal{tr}((y^{(i)}-Wx^{(i)})^T(y^{(i)}-Wx^{(i)}))$

$\frac{\partial}{\partial W_{ij}}\frac{1}{2}\mathcal{tr}((y^{(i)}-Wx^{(i)})^T(y^{(i)}-Wx^{(i)}))$

$\space\space\space\space\space= \frac{\partial}{\partial W_{ij}}\frac{1}{2}\sum_{i=1}^{n}((y^{(i)}-Wx^{(i)})^T(y^{(i)}-Wx^{(i)}))$

$\space\space\space\space\space= \sum_{i=1}^{n}(x^{(i)}(y^{(i)}-Wx^{(i)})^T) - \sum_{i=1}^{n}(x^{(i)}(y^{(i)}-Wx^{(i)}))$

$\space\space\space\space\space= \sum_{i=1}^{n}(x^{(i)}(y^{(i)}-Wx^{(i)}))$

To minimize the above expression, we set the gradient to zero adn solve for $W$:

$\sum_{i=1}^{n}(x^{(i)}(y^{(i)}-Wx^{(i)})) = 0$

$\sum_{i=1}^{n}x^{(i)}y^{(i)}-W(\sum_{i=1}^{n}x^{(i)}x^{(i)^T}) = 0$

$x^Ty = W(x^Tx)$

$W = (x^Tx)^-1(x^Ty)$

## Regularized Least Squares

In lecture, we worked through the following least squares problem

$$
\arg\min_{\theta}\frac{1}{2}\sum_{i=1}^{N}(y^{(i)}-\theta^{T}\hat{x}^{(i)})^{2}
$$

However, the least squares has a tendency to overfit the training data. One common technique
used to address the overfitting problem is regularization. In this problem, we work through
one of the regularization techniques namely ridge regularization which is also known as the
regularized least squares problem. In the regularized least squares we solve the following
optimization problem

$$
\arg\min_{\theta}\frac{1}{2}\sum_{i=1}^{N}(y^{(i)}-\theta^{T}\hat{x}^{(i)})^{2}+\frac{\lambda}{2}\Vert{\theta}\Vert_{2}^{2}
$$

where $\lambda$ is a tunable regularization parameter. From the above cost function it can be
observed that we are seeking least squares solution with a smaller 2-norm. Derive the solution
to the regularized least squares problem, i.e Find $\theta^{*}$.

**SOLUTION:**

The optimazation problem becomes:

$$
\arg\min_{\theta}\frac{1}{2}\sum_{i=1}^{N}(y^{(i)}-\theta^{T}\hat{x}^{(i)})^{2}+\frac{\lambda}{2}\Vert{\theta}\Vert_{2}^{2} = \arg\min_{\theta}\frac{1}{2}(y^{(i)}-\hat{x}^{(i)}\theta)^T(y^{(i)}-\hat{x}^{(i)}\theta)+\frac{\lambda}{2}\theta^{T}\theta
$$

Simplifing the cost function:

$$
\mathcal{L}(\theta) = \frac{1}{2}[y^{(i)^T}y^{(i)}-2y^{(i)^T}\hat{x}^{(i)}\theta+\theta^T\hat{x}^{(i)^T}\hat{x}^{(i)}\theta]+\frac{\lambda}{2}\theta^{T}\theta
$$

The Cost function is convex, so we can solve for $\theta^{*}$ by setting the dervative equal to zero:

$\nabla_{\theta}\mathcal{L}(\theta)= -\nabla_{\theta}[y^{(i)^T}\hat{x}^{(i)}\theta] + \frac{1}{2}\nabla_{\theta}[\theta^T\hat{x}^{(i)^T}\hat{x}^{(i)}\theta]+\frac{\lambda}{2}\nabla_{\theta}[\theta^{T}\theta]$

$\space\space\space\space\space= \hat{x}^{(i)^T}y^{(i)} + (\hat{x}^{(i)^T}\hat{x}^{(i)}+\lambda I)\theta$

Setting the derivative to zero and soving for $\theta^{*}$:

$\theta^{*} = (\hat{x}^{(i)^T}\hat{x}^{(i)}+\lambda I)^{-1}\hat{x}^{(i)^T}y^{(i)}$

## Linear Regression
