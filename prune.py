import numpy as np
import timeit
import ep
from sklearn.model_selection import train_test_split
import matplotlib. pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings("ignore")
import sys
print(sys.version)
def easyclass(x, y, per=0.1):
    pos = np.where(y == 1)[0]
    neg = np.where(y == -1)[0]
    xpos = x[pos]
    xneg = x[neg]

    mu_pos = np.mean(xpos, axis=0)
    mu_neg = np.mean(xneg, axis=0)
    var_pos = np.var(xpos, axis=0)
    var_neg = np.var(xneg, axis=0)

    mu = mu_pos - mu_neg
    var = 0.5 * (var_pos + var_neg)

    corr = mu/var
    corrmax = np.max(corr)
    corrmin = np.min(corr)
    if corrmax > np.abs(corrmin):
        u = np.where(corr >= corrmax*(1-per))[0]
        l = np.where(corr <= -corrmax*(1-per))[0]
    else:
        u = np.where(corr >= -corrmin*(1-per))[0]
        l = np.where(corr <= corrmin*(1-per))[0]
    #u = np.where(corr >= corrmax*(1-per))[0]
    #l = np.where(corr <= corrmin*(1-per))[0]
    inds = np.hstack((l,u))

    sortinds = np.argsort(corr[inds])

    def f(x):
        return np.dot(x[:,inds], corr[inds])

    print(x[:,inds], corr[inds])
    return inds[sortinds], corr[inds][sortinds], f

datasets = ["leukemia", "prostate", "colon"]

set = 2

x = np.genfromtxt('data/' + datasets[set] + '_x.csv', delimiter=",")
y = np.genfromtxt('data/' + datasets[set] + '_y.csv', delimiter=",")
y = y.reshape(y.shape[0],1)

n,d = x.shape

print("x:", x.shape)
print("y:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
pos = np.where(y == 1)[0]
neg = np.where(y == -1)[0]

inds, corr, f = easyclass(x_train, y_train, per=0.1)
print("Features with max abs correlation as calculated by easyclass:")
for i in range(len(inds)):
    print("\t" + str(inds[i]) + " -> " + str(corr[i]))

x_reduced = x[:,inds]
y_reduced = y
print("x_reduced.shape", x_reduced.shape)

#classification
print('classifications results')
print(f(x_test))
print(y_test)

# Plot whole array
fig = plt.figure()
plt.imshow(np.vstack((x[pos], x[neg])), interpolation='None', aspect='auto')
plt.axhline(y=len(pos), alpha=0.5, linewidth=2, color='#FF0000')
plt.colorbar()
plt.suptitle("Dataset " + datasets[set])
fig.canvas.set_window_title("Dataset " + datasets[set])

# Plot snippets of array where p is relevant for prediction (>threshold)
for i in inds:
    upper_i = max(0, min(d, i + 6))
    lower_i = max(0, min(d, i - 5))
    fig, ax = plt.subplots()
    plt.suptitle("Dataset " + datasets[set] + ": Feature " + str(i))
    fig.canvas.set_window_title("Dataset " + datasets[set] + ": Feature " + str(i))
    plt.imshow(np.vstack((x[pos,lower_i:upper_i], x[neg,lower_i:upper_i])), interpolation='None', aspect='auto')
    plt.axhline(y=len(pos), alpha=0.5, linewidth=2, color='#FF0000')
    plt.colorbar()

    ticklabels = list(range(lower_i,upper_i))
    ax.set_xticks(list(range(0,11)))
    ax.set_xticklabels(ticklabels)

plt.show()

iters = 10
k = 5
model_svc = SVC(max_iter=100000000)
scores_svc_before = np.zeros(shape=(iters,k))
scores_svc_after = np.zeros(shape=(iters,k))

def f1():
    return cross_val_score(model_svc, x_reduced, y_reduced, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
def f2():
    return cross_val_score(model_svc, x, y, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
def f3():
    easyclass(x_train, y_train, per=0.2)

#s1 = timeit.timeit(f1, number=1000)
#s2 = timeit.timeit(f2, number=1000)
# s3 = timeit.timeit(f3, number=1000)
# print(s3)
#print("Timeit1: ", s1, "\nTimeit2: ", s2)

# for i in range(iters):
#     scores_svc_after[i] = cross_val_score(model_svc, x_reduced, y_reduced, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
#     scores_svc_before[i] = cross_val_score(model_svc, x, y, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
#
# print("scores_svc before: " , scores_svc_before, "\n\tmean:", np.mean(scores_svc_before), "\n\tvar:", np.std(scores_svc_before))
# print("scores_svc after: " , scores_svc_after, "\n\tmean:", np.mean(scores_svc_after), "\n\tvar:", np.std(scores_svc_after))