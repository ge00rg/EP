import numpy as np

x = np.random.uniform(size=(5,100))
inc = 10.0
x[0, 10] += inc
x[2, 10] += inc
x[4, 10] += inc
x[0, 20] += inc
x[2, 20] += inc
x[4, 20] += inc
x[0, 30] += inc
x[2, 30] += inc
x[4, 30] += inc
x[0, 40] += inc
x[2, 40] += inc
x[4, 40] += inc
x[0, 50] += inc
x[2, 50] += inc
x[4, 50] += inc
x[0, 60] += inc
x[2, 60] += inc
x[4, 60] += inc
x[0, 70] += inc
x[2, 70] += inc
x[4, 70] += inc
x[0, 80] += inc
x[2, 80] += inc
x[4, 80] += inc
x[0, 90] += inc
x[2, 90] += inc
x[4, 90] += inc

y = np.array([1, -1, 1, -1, 1])
y = np.reshape(y, (5,1))

np.savetxt("data/X_train.txt", x)
np.savetxt("data/Y_train.txt",y)