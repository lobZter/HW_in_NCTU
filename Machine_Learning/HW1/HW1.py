import numpy as np
import math
import matplotlib.pyplot as plt

alpha = 0.1
n = 5

with open("test_data.txt") as f:
    lines = f.readlines()
lines = [line.strip() for line in lines]
a = [float(line.split(",")[0]) for line in lines]
b = [float(line.split(",")[1]) for line in lines]
# print a
# print b

plt.plot(a, b, 'ro')


k = len(lines)

A = np.empty((k, n)) #A[k][n]
for i in xrange(k):
    for j in xrange(n):
        A[i][j] = math.pow(a[i], j)

# print A

# x = (ATA+lambdaI)-1ATb
matrix_to_inverse = np.dot(np.transpose(A), A) + (alpha * np.identity(n))
# print matrix_to_inverse.shape

x = np.dot(np.dot(np.linalg.inv(matrix_to_inverse), np.transpose(A)), b)
#print x

rr = np.arange(150, 450)
plt.plot(rr, np.poly1d(x[::-1])(rr), '-')
#plt.show()

e = np.dot(A, x) - b
e = e * e
print sum(e)