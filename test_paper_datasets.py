import numpy as np
import ep
from sklearn.model_selection import train_test_split

datasets = ["test", "alon", "borovecki", "burczynski", "chiaretti", "chin", "chowdary", "christensen", "golub", "gordon", "gravier", "khan", "nakayama", "pomeroy", "shipp", "singh", "sorlie", "su", "subramanian", "sun", "tian", "west", "yeoh"]

set = 1

x = np.genfromtxt('data/' + datasets[set] + '_x.csv', delimiter=",")
y = np.genfromtxt('data/' + datasets[set] + '_y.csv', delimiter=",")
y = y.reshape(y.shape[0],1)

print("x:", x.shape)
print("y:", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0) 

f = ep.ep(x_train, y_train, 0.000000000001, 1.0, tolerance=10e-18, rho=0.5,maxiter=1000000, verbose=False)

for i in range(x_test.shape[0]):
    print(str(i) + ":", "Prediction: " + str(f(x_test[i])) + "\tLabel: " + str(y_test[i]))
