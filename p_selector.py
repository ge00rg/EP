import numpy as np
import ep

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

import matplotlib. pyplot as plt

datasets = ["leukemia", "prostate", "srbct", "colon", "adenocarcinoma"]
set = 1
threshold = 0.8

print("Running EP with dataset " + datasets[set] + "...")

x = np.genfromtxt('data/' + datasets[set] + '_x.csv', delimiter=",")
y = np.genfromtxt('data/' + datasets[set] + '_y.csv', delimiter=",")

print("x:", x.shape)
print("y:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)
pos = np.where(y == 1)[0]
neg = np.where(y == -1)[0]

print("pos: " , pos)
print("neg: " , neg)

out = ep.ep(x_train, y_train, 0.000000000001, 1.0, tolerance=10e-12, rho=0.04, maxiter=10, verbose=False)
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
inds = np.where(p > threshold)[0]

print("inds8: " , inds8)
print("inds7: " , inds7)
print("inds6: " , inds6)
print("inds5: " , inds5)
print("inds4: " , inds4)

print("inds: ", inds)
x_reduced = x[:,inds]
y_reduced = y
print("x_reduced", x_reduced.shape)

iters = 10
k = 3
model_svc = SVC(max_iter=100000000)
scores_svc_before = np.zeros(shape=(iters,k))
scores_svc_after = np.zeros(shape=(iters,k))

for i in range(iters):
    scores_svc_after[i] = cross_val_score(model_svc, x_reduced, y_reduced, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
    scores_svc_before[i] = cross_val_score(model_svc, x, y, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
    
    #print("SVC: ", scores_svc[i])

#scores_svc = cross_val_score(model_svc, x_reduced, y_reduced, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
print("scores_svc before: " , scores_svc_before, "\n\tmean:", np.mean(scores_svc_before), "\n\tvar:", np.std(scores_svc_before))
print("scores_svc after: " , scores_svc_after, "\n\tmean:", np.mean(scores_svc_after), "\n\tvar:", np.std(scores_svc_after))
