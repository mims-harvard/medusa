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

_DEF_ALPHA = 0.5
_DEF_Q = 0.25

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


def _prob(nCkq, C, s0, k, kq, ns0, weights):
    """Compute probability for an observation under a null model."""
    nCs0_q = _op(C[s0, :][:, kq] * weights[s0, None])
    nCs0 = _op(C[s0[:ns0], :])
    # at most how much can be the sum of nq smallest elements given
    # the current normalization scheme in which the sum of all elements
    # is equal to ndim? -> len(kq)
    # at least how much can the sum of nq largest elements given
    # the current normalization scheme in which the sum of all elements
    # is equal to ndim? -> len(kq)
    nCk = _op(C[k, :])
    nCkq -= len(kq)

    nCs0_diff_ncs = nCs0 - nCs0_q
    nCkq_diff_nsc = nCk - nCkq

    score = _binom(nCs0_q, nCkq) + _binom(nCs0_diff_ncs, nCkq_diff_nsc) - _binom(nCs0, nCk)
    score = np.exp(score)
    return score


def pvalue(nCkq, C, s0, k, kq, ns0, weights):
    """Compute p-value for an observation under a null model."""
    pval, err_bound = quad(_prob, len(kq), nCkq, args=(C, s0, k, kq, ns0, weights))
    return pval


def medusa(C, s0, nk, alpha=_DEF_ALPHA, q=_DEF_Q, return_itr2scores=False):
    """
    MEDUSA algorithm to compute k-maximally significant
    distant module.


    Parameters
    ----------
    C : ndarray
        Chained matrix relating objects E1 to objects E2.
    s0 : ndarray
        Indices of relevant objects E1.
    nk : int
        Desired size of the module. Used in submodular optimization.
    alpha : float, optional
        Concentration parameter promoting modules that are tight around original
        seed genes. Weight parameter to give higher weights to the seed objects compared
        to those that are agglomerated into the module at later iteration
        steps. Zero means seed objects are treated the same way as objects that
        were only predicted. One means predicted objects are not added to the seed
        set. 0.5 by default.
    q : float, optional
        Strength parameter. It defines the fraction of E2 objects whose associations
        are considered in computations. One means that associations with *all*
        E2 object are used. 0.25 by default.
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
    itr2scores : dict, optional
        Dict mapping iterations to computed scores.

    Examples
    --------
    >>> ndim = 40
    >>> s0 = np.arange(5)
    >>> ns0 = len(s0)
    >>> a = 0.8*np.random.rand(ns0, ns0)
    >>> cov = np.dot(a, a.T)
    >>> C = np.random.rand(100, ndim)
    >>> C[s0] = np.random.multivariate_normal(np.zeros(ns0), cov, ndim).T
    >>> S, P, _ = medusa(C, s0, 10)
    """

    # ------------- check input ----------------------------------------------
    n, ndim = C.shape
    ns0 = len(s0)

    if not nk <= n:
        raise ValueError('Size of module must be less than the number of objects.')

    if any(s0 < 0) or any(s0 >= n):
        raise ValueError('Relevant objects E1 must be provided in C.')

    if q <= 0. or q > 1:
        raise ValueError('Strength parameter must be in (0,1) interval.')

    _log.debug(
        '[Config] C: %s | S0: %d | Nk: %7.1e | alpha: %7.1e' %
        (str(C.shape), ns0, nk, alpha)
    )

    itr2scores = {} if return_itr2scores else None

    # ------- normalize C for the null model ----------------------------------
    C = C / C.sum(axis=1)[:, None] * ndim
    C = np.nan_to_num(C)
    nq = int(q * ndim)

    # ------- compute module --------------------------------------------------
    KQ = [(k, np.argsort(C[k])[-nq:]) for k in range(n) if k not in s0]
    fit = np.zeros(nk)
    weights = np.zeros(n)
    weights[s0] = 1.
    S, P = [], []
    exectimes = []
    for itr in range(nk):
        tic = time.time()

        pvalues = [
            (k, pvalue(_op(C[k, kq]), C, s0, k, kq, ns0, weights))
            for k, kq in KQ
            ]
        scores = [(k, np.exp(-pv), pv) for k, pv in pvalues]
        scores = sorted(scores, key=itemgetter(1), reverse=True)

        KQ = [(k, kq) for k, kq in KQ if k != scores[0][0]]
        S.append(scores[0][0])
        P.append(scores[0][2])

        if return_itr2scores:
            itr2scores[itr] = scores

        if scores[0][0] not in s0:
            weights[scores[0][0]] = (1. - alpha)**(itr + 1)
            s0 = np.r_[s0, scores[0][0]]

        fit[itr] = fit[itr-1] + scores[0][1]

        toc = time.time()
        exectimes.append(toc - tic)

        # _log.info('[%3d] fit: %0.5f | object: %d | p-value: %6.3e | '
        #           'secs: %.5f | s_itr: %s' % (
        #     itr, fit[itr], S[-1], P[-1], exectimes[-1], ', '.join(map(str, s0))
        # ))

        _log.info('[%3d] fit: %0.5f | object: %d | p-value: %6.3e | secs: %.5f' % (
            itr, fit[itr], S[-1], P[-1], exectimes[-1]
        ))

    _log.info('[end] non-monotone: %d' % np.sum(np.diff(fit) < 0))

    if not return_itr2scores:
        return np.array(S), np.array(P), np.array(exectimes)
    else:
        return np.array(S), np.array(P), np.array(exectimes), itr2scores
