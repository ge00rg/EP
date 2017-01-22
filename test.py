import ep
import numpy as np

#x_test  =np.array([[0.1,0.0],[0.2,0.0],[0.3,0.0],[0.4,0.0],[0.5,0.0],[0.6,0.0],[0.7,0.0],[0.8,0.0],[0.9,0.0]])
#y_test  =np.array([1,1,1,1,-1,-1,-1,-1,-1])

x = np.random.uniform(size=(5,1000))
inc = 5.0
x[0, 200] += inc
x[2, 200] += inc
x[4, 200] += inc
x[0, 400] += inc
x[2, 400] += inc
x[4, 400] += inc
x[0, 600] += inc
x[2, 600] += inc
x[4, 600] += inc
x[0, 800] += inc
x[2, 800] += inc
x[4, 800] += inc

y = np.array([1, -1, 1, -1, 1])
y = np.reshape(y, (5,1))

mu, v = ep.ep(x, y, 0.000000000001, 1.0, tolerance=10e-18, rho=0.1)
print(np.where(v > 0.5))
