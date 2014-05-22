# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import numpy as np

import hftools.plotting
from hftools.constants import k
from hftools.dataset import DimMatrix_j, DimMatrix_i, DataBlock
from hftools.networks.multiports import SArray, YArray, ABCDArray, ZArray,\
    convert
from hftools.math import matrix_multiply


def Hconj(A):
    i_idx = A.dims_index("i")
    j_idx = A.dims_index("j")

    order = list(range(len(A.shape)))
    order[i_idx], order[j_idx] = order[j_idx], order[i_idx]
    out = A.transpose(*order)
    dims = list(out.dims)
    dims[i_idx] = DimMatrix_i(dims[i_idx], name="i")
    dims[j_idx] = DimMatrix_j(dims[j_idx], name="j")
    out.dims = dims
    out2 = out.conj()
    return out2


def passive_noise(twoport, Tamb=290):
    if isinstance(twoport, YArray):
        C = 2 * k * Tamb * (twoport + Hconj(twoport))
    elif isinstance(twoport, ZArray):
        C = 2 * k * Tamb * (twoport + Hconj(twoport))
    elif isinstance(twoport, SArray):
        eye = hftools.dataset.make_matrix(np.array([[1, 0], [0j, 1]]),
                                          dims=tuple())
        C = k * Tamb * (eye - matrix_multiply(twoport, Hconj(twoport)))
    else:
        raise Exception("Can only convert passive Y, Z, and S to NoisyTwoport")
    return NoisyTwoport(twoport, C)


def from_noisepar(dataset):
    dataset.S = SArray(dataset.S)
    dataset.ABCD = ABCDArray(dataset.S)
    dataset.Yopt = 1 / 50. * (1 - dataset.Gopt) / (1 + dataset.Gopt)
    dataset.CAm = np.zeros_like(dataset.S)
    dataset.CAm11 = dataset.Rn
    dataset.CAm21 = 0.5 * (dataset.Fmin - 1) - dataset.Rn * dataset.Yopt
    dataset.CAm12 = dataset.CAm21.conj()
    dataset.CAm22 = dataset.Rn * abs(dataset.Yopt) ** 2
    dataset.CAm = 4 * k * 290 * dataset.CAm
    N = NoisyTwoport(dataset.ABCD, dataset.CAm)
    return N


class NoisyTwoport(object):
    def __init__(self, twoport, corrmatrix):
        self.N = twoport
        self.C = corrmatrix

    def convert(self, dest, Tout=False):
        N = dest(self.N)
        _, T_s2d, _ = convert(self.N.P, N.P, self.N)
        C = matrix_multiply(matrix_multiply(T_s2d, self.C), Hconj(T_s2d))
        C[..., 0, 0] = C[..., 0, 0].real
        C[..., 1, 1] = C[..., 1, 1].real
        if Tout:
            return NoisyTwoport(N, C), T_s2d
        else:
            return NoisyTwoport(N, C)

    def T(self, dest):
        N = dest(self.N)
        _, T_s2d, _ = convert(self.N.P, N.P, self.N)
        return T_s2d

    def __repr__(self):
        return "NoisyTwoport(%r)" % self.N.__class__.__name__

    def noise_parameters(self, partial=False):
        db = DataBlock()
        db.S = SArray(self.N)
        if isinstance(self.N, ABCDArray):
            CA = self.C / (2 * k * 290)
            CA11 = CA[..., 0, 0]
            CA12 = CA[..., 0, 1]
            CA22 = CA[..., 1, 1]
            Rn = CA11.real
            Yopt = (np.sqrt(CA22 / CA11 - (CA12.imag / CA11) ** 2) +
                    1j * (CA12.imag / CA11))
            Fmin = 1 + (CA12 + CA11 * Yopt.conj()).real
            if partial:
                db.CA = CA
            db.Rn = Rn
            db.Yopt = Yopt
            db.Fmin = Fmin
            db.Gopt = (1 / 50. - Yopt) / (1 / 50. + Yopt)
            return db

        N = self.convert(YArray)
        Y11 = N.N[..., 0, 0]
        Y21 = N.N[..., 1, 0]
        C = N.C / (4 * k * 290)
        CY11 = C[..., 0, 0]
        CY12 = C[..., 0, 1]
        CY22 = C[..., 1, 1]
        Rn = (CY22 / abs(Y21) ** 2).real
        Ycor = Y11 - CY12 / CY22 * Y21
        Gn = (CY11 - abs(Y11 - Ycor) ** 2 * Rn).real
        Yopt = np.sqrt(Gn / Rn + Ycor.real ** 2) - 1j * Ycor.imag
        Fmin = 1 + 2 * Rn * (Ycor.real + Yopt.real)

        db.CY = C
        db.Gn = Gn
        db.Ycor = Ycor
        db.S = SArray(self.N)
        db.Rn = Rn
        db.Fmin = Fmin
        db.Gopt = (1 / 50. - Yopt) / (1 / 50. + Yopt)
        db.Yopt = Yopt
        return db

