#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os

import numpy as np
import hftools.dataset.dataset as dset

from hftools.dataset import hfarray, DimSweep, DimRep, DimMatrix_i,\
    DimMatrix_j, change_dim, DiagAxis
from hftools.testing import TestCase, make_load_tests
from hftools.dataset import DataBlock
from hftools.dataset.comments import Comments

basepath = os.path.split(__file__)[0]
load_tests = make_load_tests(dset)

VA = hfarray


class Test_changedim(TestCase):
    def setUp(self):
        self.d = DataBlock()
        self.d.P = VA([0.5, -.3, .5])
        Sdims = (DimMatrix_i("i", 2), DimMatrix_j("j", 2))
        self.d.S = VA([[11, 12], [21, 22]], dims=Sdims)
        Wdims = (DimSweep("g", 1), DimMatrix_i("i", 2), DimMatrix_j("j", 2))
        self.d.W = VA([[[11, 12], [21, 22]]], dims=Wdims)

    def test_1(self):
        change_dim(self.d, DimRep, DiagAxis)
        self.assertIsInstance(self.d.ivardata["i"], DimMatrix_i)
        self.assertIsInstance(self.d.ivardata["j"], DimMatrix_j)
        self.assertIsInstance(self.d.ivardata["g"], DimSweep)

    def test_2(self):
        change_dim(self.d, DimSweep, DiagAxis)
        self.assertIsInstance(self.d.ivardata["i"], DimMatrix_i)
        self.assertIsInstance(self.d.ivardata["j"], DimMatrix_j)
        self.assertIsInstance(self.d.ivardata["g"], DiagAxis)


class Test_interpolate(TestCase):
    def setUp(self):
        self.d = DataBlock()
        f1 = DimSweep("freq", [1e9, 2e9, 3e9, 4e9, 5e9])
        self.d.y = hfarray([1., 1.1, 1.2, 1.3, 1.4], dims=(f1,))

    def test_none(self):
        fx = DimSweep("freq", [1e9, 5e9])
        D = self.d.interpolate(fx)
        self.assertAllclose(D.y, hfarray([1., 1.4], dims=(fx,)))

    def test_none_2(self):
        fx = DimSweep("freq", [1e9, 5e9])
        D = self.d.interpolate(hfarray(fx))
        self.assertAllclose(D.y, hfarray([1., 1.4], dims=(fx,)))

    def test_none_raise(self):
        fx = DimSweep("freq", [1.5e9, 5e9])
        self.assertRaises(ValueError, self.d.interpolate, fx)

    def test_linear(self):
        fx = DimSweep("freq", [1e9, 1.5e9])
        D = self.d.interpolate(fx, "linear")
        self.assertAllclose(D.y, hfarray([1., 1.05], dims=(fx,)))

    def test_shape_error(self):
        dims = (DimSweep("freq", 2), DimSweep("power", 3))
        x = hfarray(np.zeros((2, 3), ), dims=dims)
        self.assertRaises(ValueError, self.d.interpolate, x)

    def test_no_interp(self):
        """No dimension to interpolate"""
        fx = DimSweep("Vds", [1e9, 5e9])
        D = self.d.interpolate(fx)
        self.assertAllclose(D.y, self.d.y)

    def test_unknown_mode(self):
        x = DimSweep("freq", 2)
        self.assertRaises(ValueError, self.d.interpolate, x, "unknown")

if __name__ == '__main__':
        d = DataBlock()
        d.comments = Comments(["Vgs=10", "Ig=10"])
        fi = DimSweep("Freq[GHz]", 3)
        gi = DimSweep("g", 4)
        d["Vds"] = hfarray([1, 2, 3], dims=(fi,))
        d["Id"] = hfarray([1, 2, 3, 4], dims=(gi,))
        d.guess_units()
