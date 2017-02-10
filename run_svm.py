import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold


datasets = ["our_experiment", "leukemia", "prostate", "colon", "adenocarcinoma"]

set = 0

x = np.genfromtxt('data/' + datasets[set] + '_x.csv', delimiter=",")
y = np.genfromtxt('data/' + datasets[set] + '_y.csv', delimiter=",")
y = y.reshape(y.shape[0],1)

iters = 20
k = 3

model_svc = SVC(max_iter=100000000)
scores_svc_before = np.zeros(shape=(iters,k))



for i in range(iters):
    scores_svc_before[i] = cross_val_score(model_svc, x, y, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
     
print("scores_svc before: " , scores_svc_before, "\n\tmean:", np.mean(scores_svc_before), "\n\tvar:", np.std(scores_svc_before))
