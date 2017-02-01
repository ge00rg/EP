import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import ep
import ep_gene_final

class EPClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, max_iter=100, our=True, rho=0.1):
        self.max_iter = max_iter
        self.our = our
        self.rho = rho
    
    def fit(self, X, y):
        if self.our:
            self.f_ = ep.ep(X, y, 0.000000000001, 1.0, tolerance=10e-18, rho=self.rho, maxiter=self.max_iter, verbose=False)
        else:
            self.f_ = ep_gene_final.ep_gene(X, y, kkk=self.rho)
        return self
    
    def predict(self, X):
        y_pred_pos = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred_pos[i] = self.f_(X[i,:])
        y_pred_neg = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred_neg[i] = self.f_(X[i,:]*(-1))    
        y_pred = np.argmax([y_pred_neg, y_pred_pos], axis=0)
        y_pred[y_pred == 0] = -1
        return y_pred
    
    def score(self, X, y):
        n = 0
        p = 0
        y_pred = self.predict(X)

        for i in range(y_pred.shape[0]):
            if y_pred[i] == y[i]:
                p += 1
            else:
                n += 1
        return n / (p+n)