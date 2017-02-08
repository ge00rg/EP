import numpy as np
import ep
from sklearn.model_selection import train_test_split
import matplotlib. pyplot as plt

datasets = ["leukemia", "prostate", "srbct", "colon", "adenocarcinoma"]
set = 1

print("Running EP with dataset " + datasets[set] + "...")

x = np.genfromtxt('data/' + datasets[set] + '_x.csv', delimiter=",")
y = np.genfromtxt('data/' + datasets[set] + '_y.csv', delimiter=",")

print("x:", x.shape)
print("y:", y.shape)

print("Mean:     ", np.mean(x, axis=1))
print("Variance: ", np.var(x, axis=1))
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)
pos = np.where(y == 1)[0]
neg = np.where(y == -1)[0]

print("pos: " , pos)
print("neg: " , neg)

out = ep.ep(x_train, y_train, 0.000000000001, 1.0, tolerance=10e-12, rho=0.04, maxiter=100, verbose=False)
f = out[0]
p = out[1]
print("np.max(p): " , np.max(p))

for i in range(x_test.shape[0]):
    print(str(i) + ":", "Prediction: " + str(f(x_test[i])) + "\tLabel: " + str(y_test[i]))

print("p", p)

inds8 = np.where(p > 0.8)
inds7 = np.where(p > 0.7)
inds6 = np.where(p > 0.6)
inds5 = np.where(p > 0.5)
inds4 = np.where(p > 0.4)
inds3 = np.where(p > 0.3)
inds2 = np.where(p > 0.2)
inds1 = np.where(p > 0.1)

print("inds8: " , inds8)
print("inds7: " , inds7)
print("inds6: " , inds6)
print("inds5: " , inds5)
print("inds4: " , inds4)

plt.figure(figsize=(10,30))
plt.imshow(np.vstack((x[pos], x[neg])), interpolation='None', aspect='auto')
plt.axhline(y=len(pos), alpha=0.5)
for i in inds8:
    plt.text(i, -1, '|')
    plt.text(i, x.shape[0]+1, '|')
    bottom = i - 20
    top = i + 20
    if bottom < 0:
        bottom = 0
    if top > x.shape[1]:
        top = x.shape[1]
    for j in range(bottom,top):
        plt.text(j,-1,'|\n{0:.2f}'.format(np.mean(x[pos,j])),horizontalalignment='center')
        plt.text(j,x.shape[0]+1,'|\n{0:.2f}'.format(np.mean(x[neg,j])),horizontalalignment='center')
plt.colorbar()

plt.figure()
n, bins, patches = plt.hist(p, 50, normed=True)
plt.yscale('log', nonposy='clip')
plt.show()

#histogram x
#feature selection with different cutoffs x
#image micaroarray and highlight these x
#classification using:
#svm
#mu, v from ep
#easyclass
#use easyclass for preprocessing
