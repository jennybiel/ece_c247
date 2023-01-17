# HW1

## Linear Algebra Refresher

### a) Let $Q$ be a real orthogonal matrix

1. Q transpose and Q inverse are also orthonormal
   - Proof: $Q$ is orthonormal, which means that $Q^TQ = QQ^T = I$. Therefore, $Q^T = (Q^TQ)Q^T = IQ^T = Q^{-1}$. Additionally, since $QQ^T = I, Q^{-1}Q = I$ as well, which means $Q^{-1}$ is also orthonormal.

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

### a) A jar of coins is equally populated with two types of coins. One is type “H50” and comes up heads with probability 0.5. Another is type “H60” and comes up heads with probability 0.6.

1. You take one coin from the jar and flip it. It lands tails. What is the posterior probability that this is an H50 coin?
2. You put the coin back, take another, and flip it 4 times. It lands T, H, H, H. How likely is the coin to be type H50?
3. A new jar is now equally populated with coins of type H50, H55, and H60 (with probabilities of coming up heads 0.5, 0.55, and 0.6 respectively. You
take one coin and flip it 10 times. It lands heads 9 times. How likely is the coin to be of each possible type?

### b) Students at UCLA are from these disciplines: 15% Science, 21% Healthcare, 24% Liberal Arts, and 40% Engineering. (Each student belongs to a unique discipline.) The students attend a lecture and give feedback. Suppose 90% of the Science students liked the lecture, 18% of the Healthcare students liked it, none of the Liberal Arts students liked it, and 10% of the Engineering students liked it. If a student is randomly chosen, and the student liked the lecture, what is the conditional probability that the student is from Science

### c) Consider a pregnancy test with the following statistics.

- If the woman is pregnant, the test returns “positive” (or 1, indicating the woman is pregnant) 99% of the time.
- If the woman is not pregnant, the test returns “positive” 10% of the time.
- At any given point in time, 99% of the female population is not pregnant. What is the probability that a woman is pregnant given she received a positive test? The answer should make intuitive sense; give an explanation of the result that you find.

### d) Let $x_1,x_2,...,x_n$ be identically distributed random variables. A random vector, $x$, is defined as:<h3 align="center">$x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}$ <h3> What is $\mathbb{E}(Ax + b)$ in terms of $\mathbb{E}(x)$, given that $A$ and $b$ are deterministic?</h3>

### e) Let $cov(x) = \mathbb{E}((x-\mathbb{E}x)(x-\mathbb{E}x)^T)$. What is $cov(Ax + b)$ in terms of $cov(x)$, given that $A$ and $b$ are deterministic?

## Multivariate Derivatives
