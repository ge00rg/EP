from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import SVC
from EPEstimator import EPClassifier

import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

sets = ["leukemia", "prostate", "srbct", "colon", "adenocarcinoma"]
#sets = ["srbct", "adenocarcinoma"]

for j in range(len(sets)):
    print("Reading from dataset " + sets[j] + "...")
    x = np.genfromtxt('data/' + sets[j] + '_x.csv', delimiter=",")
    y = np.genfromtxt('data/' + sets[j] + '_y.csv', delimiter=",")
    
    print("Mean:     ", np.mean(x, axis=1))
    print("Variance: ", np.var(x, axis=1))
        
    print(x.shape)
    print(y.shape)    
    model_svc = SVC(max_iter=100000000)
    
    iters = 1
    k = 3
    rhos = [0.04, 0.05, 0.1, 0.2]
    
    scores_ep = np.zeros(shape=(iters,k))
    scores_ep_gene = np.zeros(shape=(iters,k))
    scores_svc = np.zeros(shape=(iters,k))
    text_file = open("results/cross_validation_" + sets[j] + ".txt", "a")
    text_file.write("================CROSS VALIDATION================\n")
    text_file.write("Time:\t %s\nDataset:\t %s\n\n"% (str(datetime.now()), sets[j]))
    text_file.flush()
    for i in range(iters):
        scores_svc[i] = cross_val_score(model_svc, x, y, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
        print("SVC: ", scores_svc[i])
    text_file.write("Results sklearn-SVC:\n%s\n\tMean:%s\n\tStd:%s\n" % (scores_svc, np.mean(scores_svc),np.std(scores_svc)))
    text_file.flush()
    
    for r in rhos:
        text_file.write("Rho = %s\n" % r)
        model_ep_our = EPClassifier(max_iter = 100, our=True, rho=r)
        for i in range(iters):
            scores_ep[i] = cross_val_score(model_ep_our, x, y, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
            print("Our EP: ", scores_ep[i])
        text_file.write("Our EP:\n%s\n\tMean:%s\n\tStd:%s\n" % (scores_ep, np.mean(scores_ep),np.std(scores_ep)))
        text_file.flush()
        
        model_ep_gene = EPClassifier(max_iter = 100, our=False, rho=r)
        for i in range(iters):
            scores_ep_gene[i] = cross_val_score(model_ep_gene, x, y, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
            print("Gene EP: ", scores_ep_gene[i])
        text_file.write("EP Gene:\n%s\n\tMean:%s\n\tStd:%s\n" % (scores_ep_gene, np.mean(scores_ep_gene),np.std(scores_ep_gene)))
        text_file.write("------------------------------------------------\n")
        text_file.flush()
    text_file.close()    