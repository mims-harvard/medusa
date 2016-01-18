# coding: utf-8
# Copyright (C) 2016 Marinka Zitnik <marinka@cs.stanford.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from operator import itemgetter
import logging
import time

import numpy as np
from scipy.special import gammaln
from scipy.integrate import quad


__version__ = '0.1'
__all__ = ['medusa']

_DEF_BETA = 0.05

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('MEDUSA')


def _binom(n, k):
    """Continuous binomial coefficient. It provides the functional
    form necessary to interpolate Newton's generalized binomial coefficient."""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _op(X):
    """Compute power of an object profile segment."""
    score = np.sum(X)
    return score


def _prob(nCs, C, s0, k, f, ndim):
    """Compute probability for an observation under a null model."""
    idxr = np.arange(C.shape[0])
    idxr = np.delete(idxr, f)

    nC = _op(C[idxr, :])
    nCk = _op(C[k, :])
    nCs0 = _op(C[idxr, :][:, s0])

    idx = np.arange(C.shape[1])
    idx = np.delete(idx, s0)
    nC_diff_Cs0 = _op(C[idxr, :][:, idx])

    nCk_diff_Cs = ndim - nCs

    score = _binom(nCs0, nCs) + _binom(nC_diff_Cs0, nCk_diff_Cs) - _binom(nC, nCk)
    score = np.exp(score)
    return score


def pvalue(nCs, C, s0, k, forbidden, ndim):
    """Compute p-value for an observation under a null model."""
    pval, err_bound = quad(_prob, nCs, ndim, args=(C, s0, k, forbidden, ndim))
    return pval


def kl(C, b, k):
    """Compute similarity of two distributions based on KL-divergence."""
    ndim = C.shape[1]
    p,  q = C[b]/ndim, C[k]/ndim
    score = np.sum(np.nan_to_num([p[i] * np.log(p[i]/q[i]) for i in range(ndim)]))
    sim = np.exp(-score)
    return sim


def medusa(C, s0, nk, beta=_DEF_BETA, return_itr2scores=False):
    """
    MEDUSA algorithm to compute `nk`-maximally significant module to the pivots `s0`.


    Parameters
    ----------
    C : ndarray
        Chained matrix relating objects E1 to objects E2.
    s0 : ndarray
        Indices of relevant objects E2.
    nk : int
        Desired size of the module. Used in submodular optimization.
    beta : float, optional
        Visibility parameter promoting diverse modules. Zero means
        no regularization. 0.05 by default.
    return_itr2scores : bool, optional
        Return a dict keyed by iterations that holds computed scores of the
        objects. False by default.

    Returns
    -------
    S : ndarray
        Array of shape (nk,) holding objects in the module.
    P : ndarray
        Array of shape (nk,) holding p-values for objects in the module.
    exectimes : ndarray
        Execution times to expand the module in each iteration.
    itr2pvalues : dict, optional
        Dict mapping iterations to computed p-values.

    Examples
    --------
    >>> s0 = np.arange(5)
    >>> C = np.random.rand(100, 40)
    >>> C[50:60][:, s0] = 1
    >>> C[50] = np.copy(C[58])
    >>> C[20, s0] = 100
    >>> C[80] = 100
    >>> S, P, _ = medusa(C, s0, 10)
    """

    # ------------- check input ----------------------------------------------
    n, ndim = C.shape

    if not nk <= n:
        raise ValueError('Size of module must be less than the number of objects.')

    if any(s0 < 0) or any(s0 >= ndim):
        raise ValueError('Relevant objects E2 must be provided in C.')

    _log.debug(
        '[Config] C: %s | S0: %d | Nk: %7.1e | alpha: %7.1e' %
        (str(C.shape), len(s0), nk, beta)
    )

    itr2scores = {} if return_itr2scores else None

    # ------- normalize C for the null model ----------------------------------
    C = C / C.sum(axis=1)[:, None] * ndim
    C = np.nan_to_num(C)

    # ------- compute module --------------------------------------------------
    K = list(range(n))
    fit, weights = np.zeros(nk), np.zeros(n)
    S, P = [], []
    exectimes = []
    for itr in range(nk):
        tic = time.time()

        pvalues = [
            (k, pvalue((1. - beta * weights[k]) * _op(C[k, s0]), C, s0, k, [], ndim))
            for k in K
            ]
        scores = [(k, np.exp(-pv), pv) for k, pv in pvalues]
        scores = sorted(scores, key=itemgetter(1), reverse=True)

        K.remove(scores[0][0])
        S.append(scores[0][0])
        P.append(scores[0][2])
        fit[itr] = fit[itr-1] + scores[0][1]

        if return_itr2scores:
            itr2scores[itr] = scores

        for k in K:
            weights[k] = beta * weights[k] + kl(C, scores[0][0], k)

        toc = time.time()
        exectimes.append(toc - tic)

        _log.info('[%3d] fit: %0.5f | object: %d | p-value: %6.3e | secs: %.5f' % (
            itr, fit[itr], S[-1], P[-1], exectimes[-1]
        ))

    _log.info('[end] non-monotone: %d' % np.sum(np.diff(fit) < 0))

    if not return_itr2scores:
        return np.array(S), np.array(P), np.array(exectimes)
    else:
        return np.array(S), np.array(P), np.array(exectimes), itr2scores
