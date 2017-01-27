import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ep


class Experiment:
    def getTestData(self, nPoints: int, dim: int, beta: np.ndarray):
        x = np.random.uniform(0.0, 10.0, size=(nPoints,dim))
        newBeta = beta[:-1]
        newBeta_0 = beta[-1]
        y = [self.getTestDataLabel(newBeta,newBeta_0,xi) for xi in x ]
        return (x,np.asarray(y).reshape((nPoints,1)))

    def getTestDataLabel(self, beta: np.ndarray, beta_0: float,  x: np.ndarray):
        """
        Labels are distributed according to logit function.
        :param beta:
        :param beta_0:
        :param x:
        :return:
        """
        prob = 1.0/(1.0 +np.exp(- (beta_0 + np.dot(beta, x) )))
        if(prob <= np.random.uniform(0.0, 1.0, size = (1,)) ):
            return 0.0
        return 1.0





def main():
    experiment = Experiment()
    beta = np.array([0.9, 0.1, -3.0])
    x,y = experiment.getTestData(1000,2, beta)
    f = ep.ep(x, y, 0.000000000001, 1.0, tolerance=10e-18, rho=0.1,maxiter=400000, verbose=False)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], y)

    n = 100
    x_test, y_test = experiment.getTestData(n,2, beta)
    y_test= y_test.reshape((n))
    x_test2  =x_test
    y_test2  = -1.0*y_test
    y_pred = [f(x) for x,y in zip(x_test,y_test)]
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(x_test[:, 0], x_test[:, 1], y_pred)
    plt.show()


if  __name__ =='__main__':main()