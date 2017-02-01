from sklearn.utils.estimator_checks import check_estimator
from sklearn.svm import SVC
from EPEstimator import EPClassifier

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

datasets = ["alon", "chin", "golub", "pomeroy", "shipp", "singh", "west"]

sets = [0]
preprocessed_data = [1]

for j in range(len(sets)):
    print("Reading from dataset " + datasets[sets[j]] + "...")
    
    if preprocessed_data[j]:
        print("Data was preprocessed.")
        x = np.genfromtxt('data/' + datasets[sets[j]] + '_pp_x.csv', delimiter=",")
        y = np.genfromtxt('data/' + datasets[sets[j]] + '_pp_y.csv', delimiter=",")
    else:
        print("Data was not preprocessed.")
        x = np.genfromtxt('data/' + datasets[sets[j]] + '_x.csv', delimiter=",")
        y = np.genfromtxt('data/' + datasets[sets[j]] + '_y.csv', delimiter=",")
    
    print("Mean:     ", np.mean(x, axis=1))
    print("Variance: ", np.var(x, axis=1))
        
    model_ep_our = EPClassifier(max_iter = 100, our=True, rho=0.025)
    model_ep_gene = EPClassifier(max_iter = 100, our=False, rho=0.025)
    model_svc = SVC(max_iter=100000000)
    
    iters = 20
    k = 3
    
    scores_ep = np.zeros(shape=(iters,k))
    scores_ep_gene = np.zeros(shape=(iters,k))
    scores_svc = np.zeros(shape=(iters,k))
    text_file = open("results/prediction_tests.txt", "a")
    text_file.write("\n--------------------------------------")
    text_file.write("\nReading from dataset %s:\n"% datasets[sets[j]])

    for i in range(iters):
        scores_ep[i] = cross_val_score(model_ep_our, x, y, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
        print("Our EP: ", scores_ep[i])
    
    #for i in range(iters):
    #    scores_ep_gene[i] = cross_val_score(model_ep_gene, x, y, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')

    for i in range(iters):
        scores_svc[i] = cross_val_score(model_svc, x, y, cv=KFold(k,shuffle=True, random_state=np.random.randint(10,1000)), scoring='accuracy')
        print("SVC: ", scores_ep[i])

    text_file.write("Our EP:\n%s\n\tMean:%s\n\tStd:%s\n" % (scores_ep, np.mean(scores_ep),np.std(scores_ep)))
    text_file.write("EP Gene:\n%s\n\tMean:%s\n\tStd:%s\n" % (scores_ep_gene, np.mean(scores_ep_gene),np.std(scores_ep_gene)))
    text_file.write("SVC:\n%s\n\tMean:%s\n\tStd:%s\n" % (scores_svc, np.mean(scores_svc),np.std(scores_svc)))
    text_file.close()    