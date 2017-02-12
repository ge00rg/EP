import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import ep
from scipy import interpolate
import seaborn


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

    def getTestDataNormal(self, nPoints: int, dim: int, beta: np.ndarray, noise: float = 0.0001):
        x = np.random.uniform(0.0, 1.0, size=(nPoints,dim))
        newBeta = beta[:-1]
        newBeta_0 = beta[-1]
        y = [self.getTestDataLabelNormal(newBeta,newBeta_0,xi,noise) for xi in x ]
        return (x,np.asarray(y).reshape((nPoints,1)))

    def getTestDataLabelNormal(self, beta: np.ndarray, beta_0: float,  x: np.ndarray, noise: float = 0.1):
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
    beta = np.array([0.5, 0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, -.5])
    sparsetys = 1./np.logspace(0, 2,20)
    sigma_0 = 1./np.logspace(10, 15,20)

    nSamples = np.linspace(1,1000,20,dtype = int)
    experiment = Experiment()
    noise = 0.000001
    xt, yt = experiment.getTestDataNormal(1000, 10, beta,noise)
    figt =plt.figure()
    figt.add_subplot(1,3,1)
    plt.scatter(xt[:,0], yt)
    figt.add_subplot(1,3,2)
    plt.scatter(xt[:,1], yt)
    figt.add_subplot(1,3,3)
    plt.scatter(xt[:,2], yt)
    plt.show()

    deltaGrid = list()
    for sparsety in sparsetys :
        deltaW = list()
        for n in nSamples:

            x,y = experiment.getTestDataNormal(n,10, beta,noise)
            negatives = np.shape(np.where(y == -1)[0])[0]
            print("There are: " + str(n - negatives) + " and :" + str(negatives) +" negatives")
            f,p = ep.ep(x, y,float("1e-19"), 1.0, tolerance=10e-18, rho=sparsety, verbose=False)
            # p/= np.max(p)
            p[np.where(p <= np.argmin(p[:1]))] =0.0
            deltaW.append((np.sqrt(np.sum(np.power(beta[:-1]-p,2)))))
            print("P values: sparsety = " + str(sparsety) + " , nsamples= " + str(n))
            print(p)
            # core = np.dot(p, beta[:-1])
            # deltaW.append(core)
        deltaGrid.append(np.asarray(deltaW))


    X, Y = np.meshgrid(sparsetys, nSamples) #[1, 2,3,4, 5, 6 ,7 ,8 ,9, 10, 11]
    fig = plt.figure(figsize=(26,20))
    ax = Axes3D(fig)
    ax.set_xlabel("sparsety")
    ax.set_ylabel("nSamples")
    ax.set_zlabel("log10(delta W)")

    # ax.set_xscale('log')


    surf = ax.plot_surface(X, Y, np.log10(np.asarray(deltaGrid)), rstride=1, cstride=1, cmap = cm.get_cmap("summer"),  linewidth=0, antialiased=True)
    fig.colorbar(surf)
    plt.show()
    fig.savefig("/home/maxweule/plot_3")


if  __name__ =='__main__':main()