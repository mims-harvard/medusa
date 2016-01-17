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
from collections import defaultdict
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


def medusa(C_list, s0, nk, alpha=_DEF_ALPHA, q=_DEF_Q, return_itr2scores=False):
    """
    MEDUSA algorithm to compute k-maximally significant
    distant module.


    Parameters
    ----------
    C_list : list of ndarray
        A list of chained matrices relating objects E1 to objects E2.
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
        Array of shape (nk,) holding scores for objects in the module.
    exectimes : ndarray
        Execution times to expand the module in each iteration.
    itr2pvalues : dict, optional
        Dict mapping iterations to computed p-values.

    Examples
    --------
    >>> ndim = 40
    >>> s0 = np.arange(5)
    >>> ns0 = len(s0)
    >>> a = 0.8*np.random.rand(ns0, ns0)
    >>> cov = np.dot(a, a.T)
    >>> C = np.random.rand(100, ndim)
    >>> C[s0] = np.random.multivariate_normal(np.zeros(ns0), cov, ndim).T
    >>> S, P, _ = medusa([C], s0, 10)
    """

    # ------------- check input ----------------------------------------------
    n, ndim = C_list[0].shape
    ns0 = len(s0)

    if not nk <= n:
        raise ValueError('Size of module must be less than the number of objects.')

    if any(s0 < 0) or any(s0 >= n):
        raise ValueError('Relevant objects E1 must be provided in C.')

    if q <= 0. or q > 1:
        raise ValueError('Strength parameter must be in (0,1) interval.')

    if any([C.shape[0] != n for C in C_list]):
        raise ValueError('Dimensions should match.')

    _log.debug(
        '[Config] C: %s | S0: %d | Nk: %7.1e | alpha: %7.1e' %
        (str(C_list[0].shape), ns0, nk, alpha)
    )

    itr2scores = {} if return_itr2scores else None

    # ------- normalize C for the null model ----------------------------------
    C_list = [C / C.sum(axis=1)[:, None] * ndim for C in C_list]
    C_list = [np.nan_to_num(C) for C in C_list]
    nq = int(q * ndim)

    # ------- compute module --------------------------------------------------
    KQ_list = [[(k, np.argsort(C[k])[-nq:]) for k in range(n)] for C in C_list]
    fit = np.zeros(nk)
    weights = np.zeros(n)
    weights[s0] = 1.
    S, P = [], []
    exectimes = []
    for itr in range(nk):
        tic = time.time()

        scores_list = []
        for KQ, C in zip(KQ_list, C_list):
            pvalues = [
                (k, pvalue(_op(C[k, kq]), C, s0, k, kq, ns0, weights))
                for k, kq in KQ
                ]
            scores = [(k, np.exp(-pv), pv) for k, pv in pvalues]
            scores = sorted(scores, key=itemgetter(1), reverse=True)
            scores_list.append(scores)

        # ------- choose chain ------------------------------------------------
        s0rank_list = [np.array([i for i, (k, _, _) in enumerate(scores) if k in s0])
                       for scores in scores_list]
        pick_weights = [np.exp(-(np.max(ranks + 1.) / len(s0) - 1.)) for ranks in s0rank_list]
        pick_weights = (np.array(pick_weights)-np.min(pick_weights))\
                       /(np.max(pick_weights)-np.min(pick_weights))
        _log.info('[%3d] chain weights: %s' % (itr, ', '.join(map(str, pick_weights))))
        pick = 0
        tmp1_scores = defaultdict(list)
        for scores in scores_list:
            for k, _, pval in scores:
                tmp1_scores[k].append(pval)
        tmp1_scores = {k: np.dot(pick_weights, vals) for k, vals in tmp1_scores.iteritems()}
        tmp_scores = [(
                           k,
                           np.exp(-tmp1_scores[k]),
                           np.nan
                       )
                       for k, _ in KQ_list[0]]
        tmp_scores = sorted(tmp_scores, key=itemgetter(1), reverse=True)
        scores_list[pick] = tmp_scores
        # ------- choose chain ------------------------------------------------

        KQ_list = [[(k, kq) for k, kq in KQ if k != scores_list[pick][0][0]]
                   for KQ in KQ_list]
        S.append(scores_list[pick][0][0])
        P.append(scores_list[pick][0][1])

        if return_itr2scores:
            itr2scores[itr] = scores_list[pick]

        if scores_list[pick][0][0] not in s0:
            weights[scores_list[pick][0][0]] = (1. - alpha)**(itr + 1)
            s0 = np.r_[s0, scores_list[pick][0][0]]

        fit[itr] = fit[itr-1] + scores_list[pick][0][1]

        toc = time.time()
        exectimes.append(toc - tic)

        _log.info('[%3d] fit: %0.5f | object: %d | score: %6.3e | '
                  'secs: %.5f | s_itr: %s' % (
            itr, fit[itr], S[-1], P[-1], exectimes[-1], ', '.join(map(str, s0))
        ))

    _log.info('[end] non-monotone: %d' % np.sum(np.diff(fit) < 0))

    if not return_itr2scores:
        return np.array(S), np.array(P), np.array(exectimes)
    else:
        return np.array(S), np.array(P), np.array(exectimes), itr2scores
