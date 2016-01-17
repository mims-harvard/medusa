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

import logging
from operator import add
from collections import defaultdict

import numpy as np
import scipy.linalg as spla

try:
    from nimfa.methods import seeding
except ImportError:
    print('Nimfa is library was not found. Certain '
          'initialization algorithms will not be available.')


__version__ = '0.1'
__all__ = ['dfmf']


logging.basicConfig(level=logging.INFO)
_log = logging.getLogger('Medusa - Collective MF')


def _initialize_random(cs, ns, random_state):
    """Random initialization of latent matrices.

    Parameters
    ----------
    cs : {list-like}
        A vector of factorization ranks, one value for
        each object type.

    ns : {list-like}
        A vector of the number of objects of every object type.

    random_state : RandomState instance
        The seed of the pseudo random number generator that is
        used for initialization of latent matrices.
    """
    _log.info('[Random] Factor initialization')
    G = {'shape': (len(cs), len(cs))}
    for i, (ci, ni) in enumerate(zip(cs, ns)):
        _log.debug('[%d, %d] Factor initialization' % (ni, ci))
        G[i, i] = random_state.rand(ni, ci)
    return G


def _initialize_nimfa(cs, R, typ='random_c'):
    """Initialization of latent matrices.

    Parameters
    ----------
    cs : {list-like}
        A vector of factorization ranks, one value for each object type.

    R : dict in the format of block indices
        A collection of data matrices represented as block entries in
        higher order matrix structure (block indices).

    typ : 'random_c' | 'random_vcol' | 'nndsvd'
        The name of initialization algorithm implemented in Nimfa.
    """
    _log.info('[Nimfa - %s] Factor initialization' % typ)
    init_type = {'random_c': seeding.random_c.Random_c(),
                 'random_vcol': seeding.random_vcol.Random_vcol(),
                 'nndsvd': seeding.nndsvd.Nndsvd()}
    G = {'shape': (len(cs), len(cs))}
    for i, ci in enumerate(cs):
        tmp = [R[i, j] for j in xrange(R['shape'][0]) if (i, j) in R] + \
              [R[j, i].T for j in xrange(R['shape'][0]) if (j, i) in R]
        if not len(tmp):
            continue
        Rij = tmp[0]
        _log.debug('[%d, %d] Factor initialization' % (Rij.shape[0], ci))
        if init_type == 'random_vcol':
            Gi, _ = init_type[typ].initialize(np.mat(Rij), ci,
                    {'p_c': 0.5 * Rij.shape[1]})
        elif init_type == 'random_c':
            Gi, _ = init_type[typ].initialize(np.mat(Rij), ci,
                    {'p_c': 0.5 * Rij.shape[1], 'l_c': 0.8 * Rij.shape[1]})
        else:
            Gi, _ = init_type[typ].initialize(np.mat(Rij), ci, {})
        Gi = np.abs(Gi)
        G[i, i] = Gi + 1e-5
    return G


def _bdot(A, B):
    """Block matrix multiplication. It returns the product of
    matrices in the format of block indices.

    Parameters
    ----------
    A : dict in the format of block indices
        First input matrix structure.

    B : dict in the format of block indices
        Second input matrix structure.
    """
    q, s1 = A['shape']
    s2, r = B['shape']
    assert (s1 == s2), 'Block structure dimension mismatch.'
    C = {'shape': (q, r)}
    for i in xrange(q):
        for j in xrange(r):
            l = [np.dot(A[i, k], B[k, j]) for k in xrange(s1)
                 if (i, k) in A and (k, j) in B]
            if len(l) > 0:
                C[i, j] = reduce(add, l)
                C[i, j] = np.nan_to_num(C[i, j])
    return C


def _btranspose(A):
    """Block matrix transpose. Every matrix is transposed
    independently of others and the result is returned in the format
    of block indices.

    Parameters
    ----------
    A : dict in the format of block indices
        Input matrix structure.
    """
    At = {'shape': (A['shape'][1], A['shape'][0])}
    for k, V in A.iteritems():
        if k == 'shape':
            continue
        At[k] = V.T
    return At


def dfmf(R, Theta, ns, cs, max_iter=100, init_typ='random_c',
        target_eps=None, system_eps=None, compute_err=False,
        return_err=False, random_state=None, callback=None):
    """Data fusion by matrix factorization (DFMF) algorithm.

    DFMF takes as its input a collection of data matrices R_ij and a
    collection of constraint matrices Theta_i. It estimates the
    corresponding latent matrices, G_i and S_ij. Here, i and j
    denote object types.

    Data matrices are passed in the following format::

        R = {'shape': (r, r),
            (0, 1): R_12, ..., (0, r - 1): R_1r,
                                ...,
            (r - 1, 0): R_r1, (r - 1, 1): R_r2, ..., (r - 1, r - 2): R_rr-1},

    where r denotes the number of object types modeled by the system.

    Similarly, constraint matrices are presented in the form::

        Theta = {'shape': (r, r),
                (0, 0): [Theta_1^(1), Theta_1^(2), ...],
                                ...,
                (r - 1, r - 1): [Theta_r^(1), Theta_r^(2), ...]},

    where r is the number of object types in the system.

    Parameters
    ----------
    R : dict in the format of block indices
        A collection of data matrices represented as block entries in
        higher order matrix structure (block indices).

    Theta : dict in the format of block indices
        A collection of constraint matrices represented as block entries
        in higher order matrix structure (block indices).

    ns : {list-like}
        A vector of the number of objects of every object type.

    cs : {list-like}
        A vector of factorization ranks, one value for each object type.

    max_iter : int
        Maximum number of iterations to be performed.

    init_typ : string
        The name of the algorithm to be used for initialization of
        latent matrices. Default choice is random C algorithm.

    target_eps : tuple (identifier of target_matrix, eps)
        Stopping criteria. If reconstruction error of the target matrix
        changes by less than eps between two consecutive iterations
        then factorization algorithm is terminated.

    system_eps : float
        Factorization algorithm is terminated when the reconstruction error
        of complete data system improves by less than system_eps
        between two consecutive iterations.

    compute_err : boolean
        If true, the value of the objective function is computed after
        every iteration and reported to the user.

    random_state : RandomState instance, or None (default)
        The seed of the pseudo random number generator that is used for
        initialization of latent matrices.

    callback : callable
        A Python function or method is called after every iteration.
        Its arguments are current estimates of the latent matrices,
        i.e. backbone matrices and recipe matrices. Backbone and recipe
        matrices are provided in the format with block indices in the
        same way as input data.
    """
    if isinstance(random_state, np.random.RandomState):
        random_state = random_state
    else:
        random_state = np.random.RandomState(random_state)
    n = len(ns)
    if isinstance(init_typ, dict):
        G = init_typ
    elif init_typ != 'random':
        G = _initialize_nimfa(cs, R, init_typ)
    else:
        G = _initialize_random(cs, ns, random_state)
    S = None

    if target_eps:
        err_target = (None, None)
    if return_err or system_eps:
        err_system = (None, None)
        compute_err = True

    _log.debug('Solving for Theta_p and Theta_n')
    Theta_p, Theta_n = defaultdict(list), defaultdict(list)
    for r, thetas in Theta.iteritems():
        if r == 'shape':
            continue
        for theta in thetas:
            t = theta > 0
            Theta_p[r].append(np.multiply(t, theta))
            Theta_n[r].append(np.multiply(t-1, theta))

    obj = []
    for itr in xrange(max_iter):
        if itr > 1 and target_eps and err_target[1] - err_target[0] < target_eps[1]:
            _log.info('[%5.4f - %5.4f < %5.4f] Early termination | Target matrix' %
                       (err_target[1], err_target[0], target_eps[1]))
            break
        if itr > 1 and system_eps is not None and err_system[1] - err_system[0] < system_eps:
            _log.info('[%5.4f - %5.4f < %5.4f] Early termination | System' %
                       (err_system[1], err_system[0], system_eps))
            break

        _log.info('[%d] Iteration' % itr)

        #######################################################################
        ########################### General Update ############################

        pGtG = {'shape': (n, n)}
        for r in G:
            if r == 'shape':
                continue
            _log.debug('[%s] GrtGr:' % str(r))
            GrtGr = np.nan_to_num(np.dot(G[r].T, G[r]))
            pGtG[r] = spla.pinv(GrtGr)

        _log.info('[%d] Updating S' % itr)

        S = _bdot(pGtG, _bdot(_btranspose(G), _bdot(R, _bdot(G, pGtG))))

        #######################################################################
        ########################### General Update ############################

        _log.info('[%d] Updating G' % itr)

        G_enum = {r: np.zeros(Gr.shape) for r, Gr in G.iteritems()
                  if isinstance(r, tuple)}
        G_enum['shape'] = (n, n)

        G_denom = {r: np.zeros(Gr.shape) for r, Gr in G.iteritems()
                  if isinstance(r, tuple)}
        G_denom['shape'] = (n, n)

        for r in R:
            if r == 'shape':
                continue
            i, j = r

            tmp1 = np.dot(R[i, j], np.dot(G[j, j], S[i, j].T))
            t = tmp1 > 0
            tmp1p = np.multiply(t, tmp1)
            tmp1n = np.multiply(t-1, tmp1)

            tmp2 = np.dot(S[i, j], np.dot(G[j, j].T, np.dot(G[j, j], S[i, j].T)))
            t = tmp2 > 0
            tmp2p = np.multiply(t, tmp2)
            tmp2n = np.multiply(t-1, tmp2)

            tmp4 = np.dot(R[i, j].T, np.dot(G[i, i], S[i, j]))
            t = tmp4 > 0
            tmp4p = np.multiply(t, tmp4)
            tmp4n = np.multiply(t-1, tmp4)

            tmp5 = np.dot(S[i, j].T, np.dot(G[i, i].T, np.dot(G[i, i], S[i, j])))
            t = tmp5 > 0
            tmp5p = np.multiply(t, tmp5)
            tmp5n = np.multiply(t-1, tmp5)

            G_enum[i, i] += tmp1p + np.dot(G[i, i], tmp2n)
            G_denom[i, i] += tmp1n + np.dot(G[i, i], tmp2p)

            G_enum[j, j] += tmp4p + np.dot(G[j, j], tmp5n)
            G_denom[j, j] += tmp4n + np.dot(G[j, j], tmp5p)

        _log.info('[%d] Solving for constraint matrices' % itr)
        for r, thetas_p in Theta_p.iteritems():
            if r == 'shape':
                continue
            _log.debug('Theta positive | %s' % str(r))
            for theta_p in thetas_p:
                G_denom[r] += np.dot(theta_p, G[r])
        for r, thetas_n in Theta_n.iteritems():
            if r == 'shape':
                continue
            _log.debug('Theta negative | %s' % str(r))
            for theta_n in thetas_n:
                G_enum[r] += np.dot(theta_n, G[r])

        for r in G:
            if r == 'shape':
                continue
            G[r] = np.multiply(G[r], np.sqrt(
                np.divide(G_enum[r], G_denom[r] + np.finfo(np.float).eps)))

        #######################################################################

        if target_eps:
            target, eps = target_eps
            err_target = (np.linalg.norm(R[target] - np.dot(G[target[0], target[0]],
                    np.dot(S[target], G[target[1], target[1]].T))), err_target[0])

        if compute_err:
            s = 0
            for r in R:
                if r == 'shape':
                    continue
                i, j = r
                Rij_app = np.dot(G[i, i], np.dot(S[i, j], G[j, j].T))
                r_err = np.linalg.norm(R[r]-Rij_app, 'fro')
                s += r_err
                _log.info('[%d] relation: %s | fit: %5.4f' % (itr, str(r), r_err))
            _log.info('[%d] system | fit: %5.4f' % (itr, s))
            obj.append(s)
            if return_err or system_eps:
                err_system = (s, err_system[0])

        if callback:
            callback(G, S)

    if compute_err:
        _log.info('Objective function violation | %d' % np.sum(np.diff(obj) > 0))
    if return_err:
        return G, S, err_system[0]
    else:
        return G, S
