import numpy as np
import matplotlib.pyplot as plt
import ep

sparsity_w_a = 0.1
sparsity_w_b = 0.8
var_a = 0.001
var_x = 1

n = 50
d = 300

def f(x,a,w):
    return np.sign(np.dot(w,np.tanh(np.dot(a,x.T))))

def generate_labels(x):
    n,d = x.shape
    #w = np.random.beta(sparsity_w_a, sparsity_w_b, size=d) * (np.random.random_integers(0,1,size=d) * 2 - 1)
    w = np.random.beta(sparsity_w_a, sparsity_w_b, size=d)
    a = np.random.binomial(1,0.1,size=(d,d)) * np.random.normal(0,np.sqrt(var_a),size=(d,d))
    np.fill_diagonal(a, 1)
    return f(x,a,w)

def add_noise(x):
    n,d = x.shape
    z = np.random.normal(0,np.sqrt(var_x),size=(n,d//100))
    return np.hstack((x,z))

x = np.random.normal(0,np.sqrt(var_x),size=(n,d))
x_noised = add_noise(x)

y = generate_labels(x_noised)

np.savetxt("data/our_experiment_x.csv", x, delimiter=",")
np.savetxt("data/our_experiment_y.csv",y, delimiter=",")