import numpy as np
import ep
from sklearn.model_selection import train_test_split
import matplotlib. pyplot as plt

def easyclass(x, y, per=0.1):
    pos = np.where(y == 1)[0]
    neg = np.where(y == -1)[0]
    xpos = x[pos]
    xneg = x[neg]

    mu_pos = np.mean(xpos, axis=0)
    mu_neg = np.mean(xneg, axis=0)
    #var_pos = np.var(xpos, axis=0)
    #var_neg = np.var(xneg, axis=0)

    mu = mu_pos - mu_neg
    var = np.var(x, axis=0)

    corr = mu/var
    corrmax = np.max(corr)
    corrmin = np.min(corr)
    u = np.where(corr >= corrmax*(1-per))[0]
    l = np.where(corr <= corrmin*(1-per))[0]
    inds = np.hstack((l,u))

    def f(x):
        return np.dot(x[:,inds], corr[inds])

    print(x[:,inds], corr[inds])
    return inds, corr[inds], f

datasets = ["test", "leukemia", "colon", "alon", "borovecki", "burczynski", "chiaretti", "chin", "chowdary", "christensen", "golub", "gordon", "gravier", "khan", "nakayama", "pomeroy", "shipp", "singh", "sorlie", "su", "subramanian", "sun", "tian", "west", "yeoh"]

set = 1

x = np.genfromtxt('data/' + datasets[set] + '_x.csv', delimiter=",")
y = np.genfromtxt('data/' + datasets[set] + '_y.csv', delimiter=",")
y = y.reshape(y.shape[0],1)

print("x:", x.shape)
print("y:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
pos = np.where(y == 1)[0]
neg = np.where(y == -1)[0]

a, b, f = easyclass(x_train, y_train, per=0.2)
print(f(x_test), y_test)

out = ep.ep(x_train, y_train, 0.000000000001, 1.0, tolerance=10e-18, rho=0.04,maxiter=100, verbose=False)
f = out[0]
p = out[1]
print(np.max(p))

for i in range(x_test.shape[0]):
    print(str(i) + ":", "Prediction: " + str(f(x_test[i])) + "\tLabel: " + str(y_test[i]))

inds8 = np.where(p > 0.8)
inds7 = np.where(p > 0.7)
inds6 = np.where(p > 0.6)
inds5 = np.where(p > 0.5)
inds4 = np.where(p > 0.4)
inds3 = np.where(p > 0.3)
inds2 = np.where(p > 0.2)
inds1 = np.where(p > 0.1)

print(inds8, inds7, inds6, inds5, inds4)

plt.figure(figsize=(10,30))
plt.imshow(np.vstack((x[pos], x[neg])), interpolation='None', aspect='auto', cmap='viridis')
plt.axhline(y=len(pos), alpha=0.5)
#for i in inds8[0]:
#    plt.text(i, -1, '|')
#    plt.text(i, x.shape[0]+1, '|')
    #plt.axvline(x=i, linestyle='dotted', alpha=0.5, color='g')
for k, i in enumerate(p):
    plt.axvline(x=k, alpha=5*i, color='k')
plt.colorbar()

plt.figure()
n, bins, patches = plt.hist(p, 50, normed=True)
plt.show()

#histogram x
#feature selection with different cutoffs x
#image micaroarray and highlight these x
#classification using:
#svm
#mu, v from ep
#easyclass
#use easyclass for preprocessing
