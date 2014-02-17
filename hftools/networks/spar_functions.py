# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import numpy as np

from hftools.dataset import make_same_info, _DimMatrix, hfarray
from hftools.math import det, inv, matrix_multiply


def cascadeS(S1, S2, *rest):
    """Cascade arrays containing S-parameters.
    """
    S1, S2 = make_same_info(S1, S2)
    neworder = tuple([x for x in S1.dims if not isinstance(x, _DimMatrix)])
    S1 = S1.reorder_dimensions(*neworder)
    S2 = S2.reorder_dimensions(*neworder)
    denom = (1 - S1[..., 1, 1] * S2[..., 0, 0])
    s1det = det(S1)
    s2det = det(S2)
    maxshape = tuple(max(x) for x in zip(S1.shape, S2.shape))
    res = S1.__class__(np.zeros(maxshape, S1.dtype), dims=S1.dims)
    res[..., 0, 0] = S1[..., 0, 0] - S2[..., 0, 0] * s1det
    res[..., 0, 1] = S1[..., 0, 1] * S2[..., 0, 1]
    res[..., 1, 0] = S1[..., 1, 0] * S2[..., 1, 0]
    res[..., 1, 1] = S2[..., 1, 1] - S1[..., 1, 1] * s2det
    res = res / denom
    res = S1.__class__(res, dims=S1.dims)
    return res


def deembedleft(e, X):
    """deembedleft deembeds S-matrix e from S-matrix X from the left
    e, and S should all be on the four-matrix form:
    Frequency is first index, measurement sweep is second index and port
    indices are third and fourth index.
    """
    X, e = make_same_info(X, e)
    neworder = tuple([x for x in X.dims if not isinstance(x, _DimMatrix)])
    X = X.reorder_dimensions(*neworder)
    e = e.reorder_dimensions(*neworder)
    klass, dims = X.__class__, X.dims
#    X, e = X.view(type=ndarray), X.view(type=ndarray)
    denom = (e[..., 0, 1] * e[..., 1, 0] -
             e[..., 0, 0] * e[..., 1, 1] +
             e[..., 1, 1] * X[..., 0, 0])
    maxshape = tuple(max(x) for x in zip(X.shape, e.shape))
    res = X.__class__(np.zeros(maxshape, X.dtype), dims=X.dims)
    e11 = (X[..., 0, 0] - e[..., 0, 0])
    e12 = (X[..., 0, 1] * e[..., 1, 0])
    e21 = (X[..., 1, 0] * e[..., 0, 1])
    e22 = (-X[..., 0, 1] * X[..., 1, 0] * e[..., 1, 1])
    res[..., 0, 0] = e11
    res[..., 0, 1] = e12
    res[..., 1, 0] = e21
    res[..., 1, 1] = e22
    res = res / denom
    res[..., 1, 1] = res[..., 1, 1] + X[..., 1, 1]
    res = klass(res, dims=dims)
    return res


def deembedright(X, e):
    """deembedright deembeds S-matrix e from S-matrix X from the right
    e, and S should all be on the four-matrix form:
    Frequency is first index, measurement sweep is second index and port
    indices are third and fourth index.
    """
    X, e = make_same_info(X, e)
    neworder = tuple([x for x in X.dims if not isinstance(x, _DimMatrix)])
    X = X.reorder_dimensions(*neworder)
    e = e.reorder_dimensions(*neworder)
    klass, dims = X.__class__, X.dims
#    X, e = X.view(type=ndarray), X.view(type=ndarray)
    denom = (e[..., 0, 1] * e[..., 1, 0] -
             e[..., 0, 0] * e[..., 1, 1] +
             e[..., 0, 0] * X[..., 1, 1])
    maxshape = tuple(max(x) for x in zip(X.shape, e.shape))
    res = X.__class__(np.zeros(maxshape, X.dtype), dims=X.dims)
    res[..., 0, 0] = (-X[..., 0, 1] * X[..., 1, 0] * e[..., 0, 0])
    res[..., 0, 1] = (X[..., 0, 1] * e[..., 1, 0])
    res[..., 1, 0] = (X[..., 1, 0] * e[..., 0, 1])
    res[..., 1, 1] = (X[..., 1, 1] - e[..., 1, 1])
    res = res / denom
    res[..., 0, 0] = res[..., 0, 0] + X[..., 0, 0]
    res = klass(res, dims=dims)
    return res


def deembed(e1, S, e2):
    """deembeds twoports, e1 and e2, at each port of S using deembed(e1,S,e2)
    e1, e2, S should all be on the four-matrix form:
    Frequency is first index, measurement sweep is second index and port
    indices are third and fourth index. Assuming same shape of all inputs
    """
    return deembedright(deembedleft(e1, S), e2)


def switch_correct(b, a):
    Sm = matrix_multiply(b, inv(a))
    return Sm


def make_passive_svd(S, delta=1e-6):
    """Force S-parameters in S to be passive.

    The S-parameters of S are scaled such that the highest eigenvalue of S'
    becomes |lambda|max < 1 - delta. The scaling is done using SVD.


    Doshi, et.al, DesignCon 2012, "Fast and Optimal Algorithms for Enforcing
    Reciprocity, Passivity and Causality in S-parameters"
    """

    svd = np.linalg.svd
    out = []
    for ss in S:
        ss = np.array(ss, copy=False)
        u, s, v = svd(ss)
        s = abs(s) - 1
        s[s < 0] = 0
        s = np.diag(s)
        delta = matrix_multiply(matrix_multiply(u, s), v)
        out.append(ss - delta)
    return S.__class__(out, dims=S.dims)


def make_passive_eig(S, delta=1e-6):
    """Force S-parameters in S to be passive.

    The S-parameters of S are scaled such that the highest eigenvalue of S'
    becomes |lambda|max < 1 - delta.
    """
    eig = np.linalg.eigvals
    out = []
    for s in S:
        lambda_max = max(abs(eig(s)))
        if lambda_max > 1:
            out.append(s / lambda_max * (1 - delta))
        else:
            out.append(s)
    return S.__class__(out, dims=S.dims)


def make_reciprocal(S):
    s12 = S[..., 0, 1]
    s21 = S[..., 1, 0]
    S12 = np.sqrt(s12 * s21)
    A12 = np.where(np.angle(S12 / s12) < 1, S12, -S12)
    S[..., 0, 1] = A12
    S[..., 1, 0] = A12
    return S


def check_passive(S):
    eig = np.linalg.eigvals
    out = []
    for s in S:
        lambda_max = max(abs(eig(s)))
        out.append(lambda_max)
    return hfarray(out, dims=S.dims[:1])

