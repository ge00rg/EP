import numpy as np
from scipy.stats import norm
import sys


def ep(x, y, var_0, var_1, rho,
       tolerance=0.0001, maxiter=100000000, verbose=True):
    '''
    x: ndarray, datapoints x features, henceforth n x d
    y: ndarray, size=n
    var_0: float, variance of spike
    var_1: float, variance of slab
    rho: float, sparsity

    tolerance: float, how much change in the parameters is accaptable
        for the approximation to be considered successful
    maxiter: int, number of iterations after which the function is stopped
    verbose: bool, if True output is generated during computing

    dimensions:
        x: (n, d)
        s: n + d + 1
        a: (n + d +1, d)
        b: (n + d +1, d)
        ny: (n + d +1, d)
        w: d
        m: (n + d + 1, d)
        p: d
        gamma: d
        mu: d
        v: d

    returns: for now, just mu and v
    '''
    n = x.shape[0]
    d = x.shape[1]

    x = np.multiply(x, y)                    # redefinition of x according to paper

    # initialize everything
    p = np.ones(d)*rho
    mu = np.zeros(d)
    v = np.ones(d)
    s = np.zeros(n + d)
    a = np.ones((n + d, d))
    b = np.ones((n + d, d))
    ny = np.ones((n + d, d))*sys.maxsize/10
    m = np.zeros((n + d, d))

    # for convergence checking
    mu_conv = np.ones(d)*sys.maxsize/10
    v_conv = np.ones(d)*sys.maxsize/10
    p_conv = np.ones(d)*sys.maxsize/10

    converge = False
    iter_count = 0

    while(not converge and iter_count < maxiter):
        i = np.random.choice(n + d)            # chose i from (0,...,1+n+d)
        # likelihood terms
        if i < n:
            p_update = False
            # 22-25
            v_old = np.reciprocal(np.reciprocal(v) - np.reciprocal(ny[i]))
            # if v_old becomes negative, go to next iteration
            if (v_old < 0).any():
                print('contined')
                continue
            mu_old = mu + np.multiply(
                np.multiply(v_old, np.reciprocal(ny[i])), mu - m[i])
            z = np.dot(
                x[i].T, mu_old)/np.sqrt(
                    np.dot(x[i].T, np.multiply(v_old, x[i])) + 1.0)
            alpha_i = (1.0/np.sqrt(
                np.dot(x[i].T, np.multiply(v_old, x[i])) + 1.0)) *\
                (norm.pdf(z)/norm.cdf(z))

            # 17-21
            mu_new = mu_old + alpha_i*np.multiply(v_old, x[i])
            v_new = v_old - (
                (alpha_i * (np.dot(x[i].T, mu_new) + alpha_i))/(
                    np.dot(x[i].T, np.multiply(v_old, x[i])) + 1.0)) *\
                np.multiply(np.multiply(v_old, x[i]), np.multiply(v_old, x[i]))
            print("v_new: ", v_new)
            print("v_old: ", v_old)
            # not sure this will be necessary later, but for now...
            if np.array_equal(v_new, v_old):
                print('cont')
                continue
            ny_new_i = np.reciprocal(
                np.reciprocal(v_new) - np.reciprocal(v_old))
            m_new_i = mu_old + np.multiply(alpha_i*ny_new_i, x[i]) +\
                np.multiply(alpha_i*v_old, x[i])
            #s_i = norm.cdf(z)*np.prod(
            #    np.sqrt((ny_new_i + v_old)/ny_new_i))*np.exp(
            #    np.sum((m_new_i - mu_old)**2/(2.0*(v_old + ny_new_i))))
            s_i = 1

            # update
            mu = mu_new
            v = v_new
            ny[i] = ny_new_i
            m[i] = m_new_i
            s[i] = s_i
        # prior terms
        else:
            p_update = True
            # 34-42
            # note: differing from the paper, our i runs from 0 to n+d, which is
            # why the indices are offsetted by n where appropriate
            v_old_i = 1.0/(1.0/v[i - n] - 1.0/ny[i, i - n])
            mu_old_i = mu[i - n] + v_old_i*(1.0/ny[i, i - n])*(mu[i - n] - m[i, i - n])
            # p_old_i = (p[i - n]/a[i, i - n])/(
            #     p[i - n]/a[i, i - n] + (1.0 - p[i - n])/b[i, i - n])
            p_old_i = rho
            G_0 = norm.pdf(0.0, loc=mu_old_i, scale=np.sqrt(v_old_i + var_0))
            G_1 = norm.pdf(0.0, loc=mu_old_i, scale=np.sqrt(v_old_i + var_1))
            Z = p_old_i*G_1 + (1 - p_old_i)*G_0
            c_1 = (1.0/Z)*(p_old_i*G_1*(-mu_old_i/(v_old_i + var_1)) +\
                (1.0 - p_old_i)*G_0*(-mu_old_i/(v_old_i + var_0)))
            c_2 = 0.5/Z*(p_old_i*G_1*(mu_old_i**2/(v_old_i + var_1)**2 -
                                      1.0/(v_old_i + var_1)) + (1 - p_old_i) *
                         G_0 * (
                             mu_old_i**2/(v_old_i + var_0)**2 -
                             1.0/(v_old_i + var_0)))
            c_3 = c_1**2 - 2.0*c_2

            # 26-33
            mu_new_i = mu_old_i + c_1*v_old_i
            v_new_i = v_old_i - c_3*(v_old_i)**2
            p_new_i = (p_old_i*G_1)/(p_old_i*G_1 + (1 - p_old_i)*G_0)
            # the next terms are all index-shifted, except s_i
            ny_new_nii = 1.0/c_3 - v_old_i
            m_new_nii = mu_old_i + c_1*(ny_new_nii + v_old_i)
            a_new_nii = p_new_i/p_old_i
            b_new_nii = (1 - p_new_i)/(1.0 - p_old_i)
            # for s_i, it seems unclear to me, what ny_new_ii is supposed to be
            s_i = Z*np.sqrt((v_old_i + ny_new_nii)/ny_new_nii)*np.exp(
                0.5*(c_1**2)/c_2**2)

            # update
            mu[i - n] = mu_new_i
            v[i - n] = v_new_i
            p[i - n] = p_new_i
            ny[i, i - n] = ny_new_nii
            m[i, i - n] = m_new_nii
            a[i, i - n] = a_new_nii
            b[i, i - n] = b_new_nii
            # for some reason the paper does not make the index correction here
            s[i] = s_i

        # convergence checking
        d_mu = np.sum(np.absolute(mu - mu_conv))
        d_v = np.sum(np.absolute(v - v_conv))
        d_p = np.sum(np.absolute(p - p_conv))

        if (d_mu + d_v + d_p <= tolerance and not d_mu + d_v + d_p == 0):
            converge = True
        print("conv :", iter_count, d_mu, d_v, d_p, d_mu + d_v + d_p, d_mu + d_v + d_p <= tolerance, p_update)

        # feedback
        if verbose and iter_count % 1000 == 0:
            print("{}th iteration, d_mu = {}, d_v = {}, dp = {}".format(
                iter_count, d_mu, d_v, d_p))
            # print(mu, v, p, ny, m, a, b, d_mu, d_v, d_p, converge)

        iter_count += 1

        mu_conv = mu.copy()
        v_conv = v.copy()
        p_conv = p.copy()

    # for now
    print(mu, v)
    return mu, v
