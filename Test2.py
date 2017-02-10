import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import ep
from hack import run


class Experiment:
    def getTestData(self, nPoints: int, dim: int, beta: np.ndarray, noise: float = 0.1):
        x = np.random.uniform(0.0, 10.0, size=(nPoints,dim))
        newBeta = beta[:-1]
        newBeta_0 = beta[-1]
        y = [self.getTestDataLabel(newBeta,newBeta_0,xi) for xi in x ]
        return (x,np.asarray(y).reshape((nPoints,1)))

    def getTestDataLabel(self, beta: np.ndarray, beta_0: float,  x: np.ndarray, noise: float = 0.1):
        """
        Labels are distributed according to logit function.
        :param beta:
        :param beta_0:
        :param x:
        :return:
        """
        prob = 1.0/(1.0 +np.exp(- (beta_0 + np.dot(beta, x) )))
        if(prob <= np.random.uniform(0.0, 1.0, size = (1,)) ):
            return -1.0
        return 1.0

    def getTestDataNormal(self, nPoints: int, dim: int, beta: np.ndarray, noise: float = 0.001):
        x = np.random.uniform(0.0, 10.0, size=(nPoints,dim))
        newBeta = beta[:-1]
        newBeta_0 = beta[-1]
        y = [self.getTestDataLabelNormal(newBeta,newBeta_0,xi,noise) for xi in x ]
        return (x,np.asarray(y).reshape((nPoints,1)))

    def getTestDataLabelNormal(self, beta: np.ndarray, beta_0: float,  x: np.ndarray, noise: float = 1.1):
        """
        Labels are distributed according to logit function.
        :param beta:
        :param beta_0:
        :param x:
        :return:
        """
        prob = np.dot(beta, x) + beta_0 + np.random.normal(0,noise)
        if(prob>= 0):
            return 1.0
        return -1.0





def main():
    beta = np.array([1.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, -5.0])
    sparsetys = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001,0.00000000001,0.000000000001,0.00000000000001]
    nSamples = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
    # sparsetys = [0.0001, 0.00001, 0.0000001, 0.000000001]
    # nSamples = [10, 40, 160, 640]

    deltaGrid = list()
    for sparsety in sparsetys :
        deltaW = list()
        for n in nSamples:
            experiment = Experiment()
            x,y = experiment.getTestDataNormal(n,10, beta)
            f,p = ep.ep(x, y, 0.001, 1.0, tolerance=10e-18, rho=sparsety, verbose=False)
            print(p)
            deltaW.append((np.sqrt(np.sum(np.power(beta[:-1]-p,2)))))
        deltaGrid.append(np.asarray(deltaW))


    print(deltaGrid)
    X, Y = np.meshgrid(np.log10(sparsetys), nSamples) #[1, 2,3,4, 5, 6 ,7 ,8 ,9, 10, 11]
    fig = plt.figure(figsize=(26,20))
    ax = Axes3D(fig)
    ax.set_xlabel("log10(sparsety)")
    ax.set_ylabel("samples")
    ax.set_zlabel("log10(delta W)")

    # ax.set_xscale('log')
    surf = ax.plot_surface(X, Y, np.log10(np.asarray(deltaGrid)), rstride=1, cstride=1, cmap = cm.get_cmap("coolwarm"),  linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()
    fig.savefig("/home/maxweule/plot1")


if  __name__ =='__main__':main()