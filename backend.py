import numpy as np

# Task a)
# Implement a method, calculating the largest eigenvector of A with b as an initial guess.
# Input: Matrix A, Vector b. A - 2D numpy array, b - 1D numpy array
# Output: The Eigenvector of A with the largest (absolute) Eigenvalue, given as 1D np.array.
def powerMethod(A: np.array, b: np.array) -> np.array:
  def normalize(x):
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n

  guess = np.ones(b.shape)

  for i in range(8):
    guess = np.dot(A, guess)
    lambda_1, guess = normalize(guess)

  return guess

# Task b)
# Implement a method, calculating the smallest eigenvector of A with b as an initial guess.
# Input: Matrix A, Vector b. A - 2D numpy array, b - 1D numpy array
# Output: The Eigenvector of A with the smallest (absolute) Eigenvalue, given as 1D np.array.
def inversePowerMethod(A: np.array, b: np.array) -> np.array:
  def normalize(x):
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n
  A_inverse = np.linalg.inv(A)
  guess = np.ones(b.shape)

  for i in range(8):
    guess = np.dot(A_inverse, guess)
    lambda_1, guess = normalize(guess)
  return guess

# ask c)
# Implement a method performing a PCA on given data.
# Input: Vectors x,y. Both 1D np.array of same size.
# Output: The Principal direction of the given data, represented as 1D np.array
def linearPCA(x: np.array, y: np.array) -> np.array:
  M = np.array((x - x.mean(), y - y.mean())).T  # creat matrix M out ot the two vectors
  Lambda, U = np.linalg.eigh(M.T.dot(M))  #getting the eigenvalues and eigenvector of covareance matrix
  index = np.argmax(abs(Lambda))
  max = U[:, index]
  return max
