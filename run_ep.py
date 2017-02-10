import numpy as np
import ep
from sklearn.model_selection import train_test_split
import matplotlib. pyplot as plt

datasets = ["our_experiment", "leukemia", "prostate", "colon"]
set = 3
rho = 0.01
threshold = 0.5

print("Running EP with dataset " + datasets[set] + "...")

x = np.genfromtxt('data/' + datasets[set] + '_x.csv', delimiter=",")
y = np.genfromtxt('data/' + datasets[set] + '_y.csv', delimiter=",")

print("x:", x.shape)
print("y:", y.shape)

n,d = x.shape
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)
pos = np.where(y == 1)[0]
neg = np.where(y == -1)[0]

f,p = ep.ep(x_train, y_train, 0.000000000001, 1.0, tolerance=10e-12, rho=rho, maxiter=100, verbose=False)

print("np.max(p): " , np.max(p))
for i in range(x_test.shape[0]):
    print(str(i) + ":", "Prediction: " + str(f(x_test[i])) + "\tLabel: " + str(y_test[i]))

mean_pos = np.mean(x[pos],axis=0)
mean_neg = np.mean(x[neg],axis=0)
var_pos = np.var(x[pos],axis=0)
var_neg = np.var(x[neg],axis=0)
corr =  (mean_pos - mean_neg) / (0.5 * (var_pos + var_neg))
mean_diff = np.abs(mean_pos - mean_neg)    

corr_max_ind = np.argpartition(np.abs(corr), -10)[-10:]
print("10 features with max abs corr: ")
for i in range(len(corr_max_ind)):
    print("\t" + str(corr_max_ind[i]) + " -> " + str(corr[corr_max_ind[i]]))

inds = np.where(p > threshold)[0]

# Plot whole array
fig = plt.figure()
plt.imshow(np.vstack((x[pos], x[neg])), interpolation='None', aspect='auto')
plt.axhline(y=len(pos), alpha=0.5, linewidth=2, color='#FF0000')
plt.colorbar()
plt.suptitle("Dataset " + datasets[set])
fig.canvas.set_window_title("Dataset " + datasets[set])
fig.savefig("results/img/" + datasets[set] + "_all_features", dpi=1200)

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

    fig.savefig("results/img/" + datasets[set] + "_feature_" + str(i), dpi=1200)

    print("Feature " + str(i) + " relevant with p=" + str(p[i]))
    mean_pos = np.mean(x[pos,i])
    mean_neg = np.mean(x[neg,i])
    mean_diff = np.abs(mean_pos - mean_neg)
    var_pos = np.var(x[pos,i])
    var_neg = np.var(x[neg,i])
    corr =  (mean_pos - mean_neg) / (0.5 * (var_pos + var_neg))
    mean_diff = np.abs(mean_pos - mean_neg)    
    print("Feature " + str(i) + " relevant with p=" + str(p[i]) + ".\n\tMean Diff: " + str(mean_diff) + "\n\tCorr: " + str(corr))
    
# Plot histogramm        
fig = plt.figure()
n, bins, patches = plt.hist(p, 50, normed=True)
plt.yscale('log', nonposy='clip')
plt.suptitle("Dataset " + datasets[set] + ": Histogramm")
fig.canvas.set_window_title("Dataset " + datasets[set] + ": Histogramm")
fig.savefig("results/img/" + datasets[set] + "_histogram", dpi=1200)

plt.show()