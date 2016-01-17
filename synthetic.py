import numpy as np

from medusa import medusa_cpe, medusa_cpi

np.random.seed(0)


def toy_cpe():
    n = 100
    ndim = 40
    s0 = np.arange(5)
    ns0 = len(s0)

    a = 0.8*np.random.rand(ns0+1, ns0+1)
    cov = np.dot(a, a.T)
    C = np.random.rand(n, ndim)
    dat = np.random.multivariate_normal(10*np.ones(ns0+1), cov, ndim).T
    # correlated profiles of the pivots
    C[s0] = dat[:len(s0)]
    idxr = np.delete(np.arange(n), s0)
    # here are candidates coming from a different data distribution than the pivots
    C[idxr] = np.random.randn(len(idxr), ndim) + 10
    # here is a candidate object coming from the same distribution as the pivots
    C[10] = dat[len(s0)]
    # two candidates with obscure profiles
    C[30] = 0
    C[30, 2] = 1.

    S, P, exectimes = medusa_cpe.medusa(C, s0, nk=10, alpha=0.7, q=0.25)
    return S, P, exectimes


def toy_cpi():
    n = 100
    ndim = 40
    s0 = np.arange(10)

    C = np.random.rand(n, ndim)
    # simulate objects with strong associations that match the pivots
    C[50:60, s0] = 1
    # simulate (near) duplicates to study module diversity
    C[50] = np.copy(C[58])
    # simulate object with very strong associations matching the pivots
    C[20, s0] = 100
    # simulate object with very strong but unspecific (relative to the pivots) associations
    C[80] = 100

    S, P, exectimes = medusa_cpi.medusa(C, s0, nk=10)
    return S, P, exectimes


toy_cpe()
toy_cpi()
