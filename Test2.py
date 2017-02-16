import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
import seaborn

import nep


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


def helper(x, idx):
    t = np.copy(x)
    t[idx] = 0
    return t



def getRank(p):
    return np.argsort(p)[::-1]

def frotman(p, dim):
    t= np.zeros(dim)
    t[0] = 4.5
    t[1] = 4.5
    t[2] = 4.5
    t[3] = 4.5
    t[4] = 4.5
    t[5] = 4.5
    t[6] = 4.5
    t[7] = 4.5
    t[8] = 4.5
    t[9] = 4.5
    t[10:] = 29.5
    res = float(np.sum(np.abs(getRank(p) - t))/float(795))
    return res

def frotman2(p, dim, aktive):
    t= np.zeros(dim)
    t[:aktive] = 1
    t[aktive:] = -1
    k = np.zeros_like(p)
    k[getRank(p)[:aktive]] = 1
    k[getRank(p)[aktive:]] = -1

    res = float(np.sum(np.abs(k - t))/float(dim))
    return res


def main():
    resolution = 20
    dim = 500
    aktive = 100
    # beta = np.array([1.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, -0.5])
    beta = np.zeros(shape=(dim + 1))
    beta[:aktive] = 1.0/aktive

    norm = frotman(range(0,500),dim)
    print (norm)
    beta[-1] = -0.5
    sparsetys = 1./np.logspace(0, 1,resolution)
    # sigma_0 = 1./np.logspace(10, 15,6)

    nSamples = np.linspace(100,1000,resolution,dtype = int)
    experiment = Experiment()
    noise = 0.000001
    xt, yt = experiment.getTestDataNormal(1000, dim, beta,noise)
    figt =plt.figure()
    figt.add_subplot(1,3,1)
    plt.scatter(xt[:,0], yt)
    figt.add_subplot(1,3,2)
    plt.scatter(xt[:,1], yt)
    figt.add_subplot(1,3,3)
    plt.scatter(xt[:,2], yt)
    plt.show()

    deltaGridDiff = list()
    deltaGridDiff2 = list()
    for sparsety in sparsetys :
        deltaWDiff = list()
        deltaWDiff2 = list()
        for n in nSamples:

            x,y = experiment.getTestDataNormal(n,dim, beta,noise)
            negatives = np.shape(np.where(y == -1)[0])[0]
            print("There are: " + str(n - negatives) + " and :" + str(negatives) +" negatives")
            f,p = nep.ep(x, y,1e-19, 1.0, tolerance=10e-18, rho=sparsety, verbose=False)
            # p/= np.max(p)
            print("P values: sparsety = " + str(sparsety) + " , nsamples= " + str(n))
            print(p)
            deltaWDiff.append((np.sqrt(np.sum(np.power(beta[:-1] - p, 2)))))
            # deltaWDiff2.append((np.sqrt(np.sum(np.power(beta[:-1] - helper(p, np.where(p < np.min(p[list([0,1,2,3,4,5,6,7,8,9])]))), 2)))))
            deltaWDiff2.append(frotman2(p,dim,aktive))
            # core = np.dot(p, beta[:-1])
            # deltaW.append(core)
        deltaGridDiff.append(np.asarray(deltaWDiff))
        deltaGridDiff2.append(np.asarray(deltaWDiff2))
    diffs = np.asarray(deltaGridDiff)
    diffs2 = np.asarray(deltaGridDiff2)
    X, Y = np.meshgrid(sparsetys, nSamples)
    fig = plt.figure(figsize=(26,20))
    fig.add_subplot(1,2,1)
    z = np.log10(np.asarray(diffs))
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    surf = plt.pcolor(X,Y,np.log10(np.asarray(diffs)),cmap = cm.get_cmap("viridis"), vmin=z_min, vmax=z_max)
    plt.axvline(0.2, color='red')
    plt.title('log absolute difference')
    plt.axis([X.min(), X.max(), Y.min(), Y.max()])
    plt.xlabel("sparsity")
    plt.ylabel("Samples")
    fig.colorbar(surf)
    # plt.yticks([int(j) for j in nSamples])
    # plt.xticks([int(j) for j in sparsetys])
    # plt.xlim(np.min(sparsetys), np.max(sparsetys))
    # fig.colorbar(surf)
    # fig.add_subplot(1, 3, 2)
    # plt.imshow(np.log10(np.asarray(diffs)[:-1, :-1]))
    #
    # # ax.set_xscale('log')



    fig.add_subplot(1, 2, 2)
    X, Y = np.meshgrid(sparsetys, nSamples) #[1, 2,3,4, 5, 6 ,7 ,8 ,9, 10, 11]
    surf = plt.pcolor(X,Y,diffs2,cmap = cm.get_cmap("viridis"))
    plt.axvline(0.2, color='red')
    plt.title('Spearman footrule Difference')
    plt.xlabel("sparsity")
    plt.ylabel("Samples")
    fig.colorbar(surf)
    plt.show()
    fig.savefig("/home/kaw/Bilder/plot_3")

if  __name__ =='__main__':main()