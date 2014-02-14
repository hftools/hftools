# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

"""
Multiports
==========

.. autoclass:: ZArray

.. autoclass:: YArray

.. autoclass:: SArray

Twoports
==========

.. autoclass:: GArray

.. autoclass:: HArray

.. autoclass:: ABCDArray

.. autoclass:: TArray

.. autoclass:: TpArray

"""
import itertools
import os
import pdb
import re
import sys

from itertools import chain

import numpy
import numpy as np
from numpy import concatenate, zeros, newaxis, array, ndarray, identity, \
    zeros_like, zeros, bmat, diag, sqrt, empty_like, asarray

from hftools.math import det, matrix_multiply, inv
from hftools.networks.spar_functions import cascadeS, deembed, deembedright,\
    deembedleft
from hftools.dataset.arrayobj import hfarray, DimSweep, DimRep,\
    DimMatrix_i, DimMatrix_j, _hfarray, ismatrix


def make_accessor(i, j):
    """Help function to define accessor for matrix elements.

    Return function that access element i, j of multi-matrix.
    i,j are zero indexed.
    """
    def accessor(x):
        """Matrix accessor function for zero indexed element (%d,%d)"""
        return hfarray(x, copy=False)[..., i, j]
    accessor.__doc__ = accessor.__doc__ % (i, j)
    return accessor


def convert(fromP, toP, a):
    """Converts multiports using permutation matrices as described in [#]

    [#] J. Stenarson and K. Yhland "" IEEE Transactions on Instrumentation and
        Measurements 2009 vol. x no. 4 pp. xx-yy.

    """
    PB = inv(toP)
    PA = fromP
    P = matrix_multiply(PA, PB)
    tau1 = P[..., :2, :2]
    sigma1 = P[..., :2, 2:]
    tau2 = P[..., 2:, :2]
    sigma2 = P[..., 2:, 2:]
    A = inv(tau1 - matrix_multiply(a, tau2))
    B = -sigma1 + matrix_multiply(a, sigma2)
    res = matrix_multiply(A, B)
    return res, A, B


class _MultiPortArray(_hfarray):
    """Basklass som ej skall anvandas direkt
    """
    shortname = None
    P = None
    _attrfuns = {}
    _Z0 = None

    def __init__(self, data, dims=None, copy=True):
        data = np.asanyarray(data)
        if (self.shortname is not None) and (data.ndim >= 2):
            self._attrfuns = {}
            names = [self.shortname.lower(), self.shortname.upper(),
                     self.shortname]
            for name in names:
                for i in range(data.shape[-1]):
                    for j in range(data.shape[-1]):
                        label = "%s%d%d" % (name, i + 1, j + 1)
                        accessor = make_accessor(i, j)
                        self._attrfuns[label] = accessor

        if isinstance(data, self.__class__):
            pass
        elif isinstance(data, _MultiPortArray):
            if self.ismatrix() and (data.shape[-1] == data.shape[-2]):
                view = data.view(type=hfarray, dtype=data.dtype)
                res, _, _ = convert(data.P, self.P, view)
                self.view(type=hfarray, dtype=self.dtype)[:] = res
            else:
                fmt = ("Can not convert subelements of a %s matrix to "
                       "elements of a %s matrix")
                msg = fmt % (data.__class__.__name__, self.__class__.__name__)
                raise ValueError(msg)

    def __new__(subtype, data, dims=None, dtype=None, copy=True, order=None,
                subok=False, ndmin=0, unit=None, info=None):

        if info is not None:
            warn("_MultiPortArray, use dims not info")
            if dims is None:
                dims = info
        info = None
        data = np.asanyarray(data)
        if dims is None:
            if hasattr(data, 'dims'):
                dims = tuple(data.dims)
            elif len(data.shape) <= len(subtype.default_dim) + 2:
                defaultdims = (subtype.default_dim)
                shape = data.shape[:-2]
                dims = tuple(x.__class__(x.name, range(size))
                             for (x, size) in zip(defaultdims, shape))
                dims = dims + (DimMatrix_i("i", data.shape[-2]),
                               DimMatrix_j("j", data.shape[-1]))
        return _hfarray.__new__(subtype, data, dims=dims, dtype=dtype,
                                   copy=copy, order=order, subok=subok,
                                   ndmin=ndmin, unit=unit)

    def ismatrix(self):
        return (self.ndim >= 2 and
                isinstance(self.info[-2], DimMatrix_i) and
                isinstance(self.info[-1], DimMatrix_j))

    def cascade(self, other):
        """Make a cascade connection of *self* and *other*.

        Converts *self* and *other* to S-parameters first if necessary.
        """
        a = SArray(self)
        b = SArray(other)
        return self.__class__(cascadeS(a, b))

    def deembed(self, left=None, right=None):
        """Deembed *left* and *right from *self*.

        Converts *self*, *left*, and *right* to S-parameters first if
        necessary.
        """
        if left is None and right is None:
            return self
        elif left is None:
            return self.__class__(deembedright(SArray(self),
                                               SArray(right)))
        elif right is None:
            return self.__class__(deembedleft(SArray(left),
                                              SArray(self)))
        else:
            return self.__class__(deembed(SArray(left),
                                          SArray(self),
                                          SArray(right)))

    def __getattr__(self, key):
        try:
            return (self._attrfuns[key](self))
        except KeyError:
            raise AttributeError

    def __setattr__(self, key, value):
        try:
            object.__getattribute__(self, key)
            object.__setattr__(self, key, value)
            return
        except AttributeError:
            pass
        self.__getattr__(key)[...] = value

    def __getitem__(self, idx):
        data = self.view(dtype=self.dtype, type=hfarray).__getitem__(idx)
        if ismatrix(data):
            return self.__class__(data, copy=False)
        else:
            return data


class _TwoPortArray(_MultiPortArray):
    """Basklass som ej skall anvandas direkt
    """

    def __init__(self, data, dims=None, copy=True):
        if data.shape[-1] != 2:
            raise ValueError("%s must be a 2x2 matrix")
        _MultiPortArray.__init__(self, data, dims, copy=copy)


def make_matrix(A):
    a = np.asanyarray(A)
    info = (DimMatrix_i("i", range(a.shape[-2])),
            DimMatrix_j("j", range(a.shape[-1])),)
    return hfarray(a, dims=info)


class ZArray(_MultiPortArray):
    """
Z-parameter model

.. aafig::
    :aspect: 80

        i1 +---------+  i2
    '+'->--+         +--<- '+'
    v1     |    Z    |     v2
    '-'----+         +---- '-'
           +---------+

     |V1|     |Z11  Z12||I1|
     |  | '=' |        ||  |
     |V2|     |Z21  Z22||I2|

This model may be of any dimension.

"""
    shortname = "Z"

    def __init__(self, data, dims=None, copy=True):
        data = np.asanyarray(data)
        self.P = make_matrix(identity(data.shape[-1] * 2))
        if isinstance(data, YArray) and data.ismatrix():
            if data.shape[-1] != data.shape[-2]:
                raise ValueError("%s must be a square matrix" % self.shortname)
            data = _hfarray(inv(data))
            self[:] = data
        _MultiPortArray.__init__(self, data, dims, copy=copy)


class YArray(_MultiPortArray):
    """
Y-parameter model

.. aafig::
    :aspect: 80

         i1 +---------+  i2
     '+'->--+         +--<- '+'
     v1     |    Y    |     v2
     '-'----+         +---- '-'
            +---------+

      |I1|     |Y11  Y12||V1|
      |  | '=' |        ||  |
      |I2|     |Y21  Y22||V2|

This model may be of any dimension.

"""
    shortname = "Y"

    def __init__(self, data, dims=None, copy=True):
        i = identity(data.shape[-1])
        o = zeros_like(i)
        self.P = make_matrix(bmat([[o, i], [i, o]]))
        if isinstance(data, ZArray):
            if data.shape[-1] != data.shape[-2]:
                raise ValueError("%s must be a square matrix" % self.shortname)
            data = _hfarray(inv(data))
            self[:] = data
        _MultiPortArray.__init__(self, data, dims, copy=copy)


class SArray(_MultiPortArray):
    """S-parameter model

.. aafig::
    :aspect: 80

              +---------+
    a1 --> ---+         +---  --> b2
              |    S    |
    b1 <-- ---+         +---- <-- a2
              +---------+

        |b1|     |S11  S12||a1|
        |  | '=' |        ||  |
        |b2|     |S21  S22||a2|

This model may be of any dimension.

"""
    shortname = "S"

    def __init__(self, data, dims=None, copy=True):
        self.Z0 = 50
        _MultiPortArray.__init__(self, data, dims, copy=copy)

    def get_Z0(self):
        return self._Z0

    def set_Z0(self, Z0):
        self._Z0 = Z0
        v = diag([sqrt(1. / Z0) / 2.] * self.shape[-1])
        i = diag([sqrt(Z0) / 2.] * self.shape[-1])
        self.P = make_matrix(bmat([[v, -i], [v, i]]))
    Z0 = property(get_Z0, set_Z0)


class GArray(_TwoPortArray):
    """G-parameter model

.. aafig::
    :aspect: 80

        i1 +---------+  i2
    '+'->--+         +--<- '+'
    v1     |    G    |     v2
    '-'----+         +---- '-'
           +---------+

      |I1|     |G11  G12||V1|
      |  | '=' |        ||  |
      |V2|     |G21  G22||I2|

This model may be of any dimension.

"""
    shortname = "G"

    def __init__(self, data, dims=None, copy=True):
        self.P = array([[0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1.]])
        self.P = make_matrix(self.P)
        _TwoPortArray.__init__(self, data, dims, copy=copy)


class HArray(_TwoPortArray):
    """H-parameter model

.. aafig::
    :aspect: 80

        i1 +---------+  i2
    '+'->--+         +--<- '+'
    v1     |    H    |     v2
    '-'----+         +---- '-'
           +---------+

      |I1|     |G11  G12||V1|
      |  | '=' |        ||  |
      |V2|     |G21  G22||I2|

This model is only valid for a twoport

"""
    shortname = "H"

    def __init__(self, data, dims=None, copy=True):
        self.P = make_matrix(array([[1, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, 0.]]))
        self.P = make_matrix(self.P)
        _TwoPortArray.__init__(self, data, dims, copy=copy)


class ABCDArray(_TwoPortArray):
    """ABCD-parameter model

.. aafig::
    :aspect: 80

        i1 +---------+  i2
    '+'->--+         +--<- '+'
    v1     |  ABCD   |     v2
    '-'----+         +---- '-'
           +---------+

       |V1|     |A    B  ||V2 |
       |  | '=' |        ||   |
       |I1|     |C    D  ||-I2|

This model is only valid for a twoport
"""
    shortname = "A"

    def __init__(self, data, dims=None, copy=True):
        self.P = array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, -1.]])
        self.P = make_matrix(self.P)
        _TwoPortArray.__init__(self, data, dims, copy=copy)


class TArray(_TwoPortArray):
    """T-parameter model

.. aafig::
    :aspect: 80

              +---------+
    a1 --> ---+         +---  --> b2
              |    T    |
    b1 <-- ---+         +---- <-- a2
              +---------+

        |b1|     |T11  T12||a2|
        |  | '=' |        ||  |
        |a1|     |T21  T22||b2|

This model may be of any dimension.

"""
    shortname = "T"

    def __init__(self, data, dims=None, copy=True):
        self.Z0 = 50
        _TwoPortArray.__init__(self, data, dims, copy=copy)

    def get_Z0(self):
        return self._Z0

    def set_Z0(self, Z0):
        self._Z0 = Z0
        Z1 = Z2 = Z0
        v1 = sqrt(1. / Z1) / 2.
        v2 = sqrt(1. / Z2) / 2.
        i1 = sqrt(Z1) / 2.
        i2 = sqrt(Z2) / 2.
        self.P = array([[v1, 0, -i1, 0],
                        [v1, 0, i1, 0],
                        [0, v2, 0, i2],
                        [0, v2, 0, -i2]])
        self.P = make_matrix(self.P)
    Z0 = property(get_Z0, set_Z0)


class TpArray(_TwoPortArray):
    a = """Tp-parameter model

.. aafig::
    :aspect: 80

              +---------+
    a1 --> ---+         +---  --> b2
              |    Tp   |
    b1 <-- ---+         +---- <-- a2
              +---------+

      |a1|     |Tp11  Tp12||b2|
      |  | '=' |          ||  |
      |b1|     |Tp21  Tp22||a2|

This model is only valid for a twoport

"""
    shortname = "Tp"

    def __init__(self, data, dims=None, copy=True):
        self.Z0 = 50
        _TwoPortArray.__init__(self, data, dims, copy=copy)

    def get_Z0(self):
        return self._Z0

    def set_Z0(self, Z0):
        self._Z0 = Z0
        Z1 = Z2 = Z0
        v1 = sqrt(1. / Z1) / 2.
        v2 = sqrt(1. / Z2) / 2.
        i1 = sqrt(Z1) / 2.
        i2 = sqrt(Z2) / 2.
        self.P = array([[v1, 0, i1, 0],
                        [v1, 0, -i1, 0],
                        [0, v2, 0, -i2],
                        [0, v2, 0, i2]])
        self.P = make_matrix(self.P)
    Z0 = property(get_Z0, set_Z0)


def unit_matrix(size=2, cls=SArray):
    i = DimMatrix_i("i", range(size))
    j = DimMatrix_j("j", range(size))
    m = hfarray(np.identity(size, dtype=np.complex128), dims=(i, j))
    return SArray(m)


def unit_smatrix(size=2, cls=SArray):
    i = DimMatrix_i("i", range(size))
    j = DimMatrix_j("j", range(size))
    m = hfarray(np.zeros((size, size), dtype=np.complex128), dims=(i, j))
    m[0, 1] = m[1, 0] = 1
    return SArray(m)


if __name__ == "__main__":
    z = ZArray(array([[1., -1], [1, 1]]))
    y = YArray(array([[1., -1], [1, 1]]))
    g = GArray(array([[1., -1], [1, 1]]))
    h = HArray(array([[1., -1], [1, 1]]))
    sshort = SArray(array([[-1., 0], [0, -1]]))
    sopen = SArray(array([[1., 0], [0, 1]]))
    zi = YArray(z)
    w = HArray(array([[1., 2], [3, 4]]))

    fi = DimSweep("freq", range(3))
    zz = ZArray([[[1., -1], [1, 1]]] * 3, dims=(fi,))

    yy = ZArray([[[1., -1, 0], [-1, 1, -1], [0, -1, 1]]] * 3, dims=(fi,))
