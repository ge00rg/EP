import numpy as np
import ep
from sklearn.model_selection import train_test_split
import matplotlib. pyplot as plt

datasets = ["test", "colon", "alon", "borovecki", "burczynski", "chiaretti", "chin", "chowdary", "christensen", "golub", "gordon", "gravier", "khan", "nakayama", "pomeroy", "shipp", "singh", "sorlie", "su", "subramanian", "sun", "tian", "west", "yeoh"]

set = 1

x = np.genfromtxt('data/' + datasets[set] + '_x.csv', delimiter=",")
y = np.genfromtxt('data/' + datasets[set] + '_y.csv', delimiter=",")
y = y.reshape(y.shape[0],1)

print(np.var(x))

print("x:", x.shape)
print("y:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
pos = np.where(y == 1)[0]
neg = np.where(y == -1)[0]

out = ep.ep(x_train, y_train, 0.000000000001, 1.0, tolerance=10e-18, rho=0.1,maxiter=100, verbose=False)
f = out[0]
p = out[1]

for i in range(x_test.shape[0]):
    print(str(i) + ":", "Prediction: " + str(f(x_test[i])) + "\tLabel: " + str(y_test[i]))

inds8 = np.where(p > 0.8)
inds7 = p[p > 0.7]
inds6 = p[p > 0.6]
inds5 = p[p > 0.5]
inds4 = p[p > 0.4]

plt.figure(figsize=(10,30))
plt.imshow(np.vstack((x[pos], x[neg])), interpolation='None', aspect='auto', cmap='viridis')
plt.axhline(y=len(pos), alpha=0.5)
for i in inds8[0]:
    plt.text(i, -1, '|')
    plt.text(i, x.shape[0]+1, '|')
    #plt.axvline(x=i, linestyle='dotted', alpha=0.5, color='g')
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
