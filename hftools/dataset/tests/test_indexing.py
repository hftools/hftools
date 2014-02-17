# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os
import sys
import unittest
import warnings

import numpy as np
from numpy import allclose, array, newaxis

import hftools

from hftools.dataset.arrayobj import hfarray, DimSweep, _hfarray,\
    DimMatrix_i, DimMatrix_j
from hftools.testing import TestCase
from hftools.utils import HFToolsWarning, reset_hftools_warnings


class TestIndex(TestCase):
    def setUp(self):
        self.I = I = DimSweep("I", [1, 2])
        self.J = J = DimSweep("J", [10, 20, 30])
        self.K = K = DimSweep("K", [100, 200, 300, 400])
        self.i, self.j, self.k = (i, j, k) = map(hfarray, (I, J, K))

        self.a = i * j * k
        self.b = hfarray(i * j * k, unit="Hz", outputformat="%.3f")
        cinfo = (DimMatrix_i("i", 2), DimMatrix_j("j", 2))
        self.c = hfarray([[11, 12], [21, 22]], dims=cinfo)

    def test_a_1(self):
        facit = array(self.a)[self.I.data == 2]
        res = self.a[self.i == 2]
        self.assertTrue(allclose(facit, res))

    def test_a_2(self):
        facit = array(self.a)[:, self.J.data == 20]
        res = self.a[self.j == 20]
        self.assertTrue(allclose(facit, res))

    def test_a_3(self):
        facit = array(self.a)[:, :, self.K.data == 200]
        res = self.a[self.k == 200]
        self.assertTrue(allclose(facit, res))

    def test_a_4(self):
        facit = array(self.a)[:, :, :-1]
        res = self.a[..., :-1]
        self.assertTrue(allclose(facit, res))

    def test_a_5(self):
        facit = array(self.a)[newaxis]
        res = self.a[newaxis]
        self.assertTrue(allclose(facit, res))
        self.assertIsInstance(res, np.ndarray)

    def test_a_6(self):
        facit = array(self.a)[newaxis]
        res = self.a[newaxis, :]
        self.assertTrue(allclose(facit, res))
        self.assertIsInstance(res, np.ndarray)

    def test_a_7(self):
        self.assertRaises(IndexError, self.a.__getitem__, (Ellipsis, Ellipsis))

    def test_a_8(self):
        def funk():
            return self.a[self.a < 3000]

        res = funk()
        self.assertIsInstance(res, np.ndarray)
        self.assertAllclose(res, [1000, 2000, 2000, 2000])


    def test_b_1(self):
        facit = array(self.b)[self.I.data == 2]
        res = self.b[self.i == 2]
        self.assertTrue(allclose(facit, res))
        self.assertEqual(res.unit, "Hz")
        self.assertEqual(res.outputformat, "%.3f")

    def test_b_2(self):
        facit = array(self.b)[:, self.J.data == 20]
        res = self.b[self.j == 20]
        self.assertTrue(allclose(facit, res))
        self.assertEqual(res.unit, "Hz")
        self.assertEqual(res.outputformat, "%.3f")

    def test_b_3(self):
        facit = array(self.b)[:, :, self.K.data == 200]
        res = self.b[self.k == 200]
        self.assertTrue(allclose(facit, res))
        self.assertEqual(res.unit, "Hz")
        self.assertEqual(res.outputformat, "%.3f")

    def test_b_4(self):
        facit = array(self.b)[:, :, :-1]
        res = self.b[..., :-1]
        self.assertTrue(allclose(facit, res))
        self.assertEqual(res.unit, "Hz")
        self.assertEqual(res.outputformat, "%.3f")

    def test_b_5(self):
        b = hfarray([1, 2, 3], unit="Hz", outputformat="%.3f")
        res = b[0]
        self.assertEqual(res, 1)
        self.assertFalse(hasattr(res, "unit"))
        self.assertFalse(hasattr(res, "outputformat"))

    def test_b_6(self):
        facit = array(self.b)[0, 0, 0]
        res = self.b[0, 0, 0]
        self.assertEqual(res, 1000)
        self.assertFalse(hasattr(res, "unit"))
        self.assertFalse(hasattr(res, "outputformat"))

    def test_c_1(self):
        self.c[0, 0] = 1
        self.assertAllclose(self.c, [[1, 12], [21, 22]])

    def test_c_2(self):
        self.c[..., 0, 0] = 1
        self.assertAllclose(self.c, [[1, 12], [21, 22]])

    def test_ellipsis_1(self):
        self.c[...] = 0
        self.assertAllclose(self.c, 0)

    def test_slice_1(self):
        a = self.a[:1]
        self.assertAllclose(array(self.a)[:1], a)
        self.assertEqual(a.dims[0], DimSweep("I", [1, 2][:1]))


if __name__ == '__main__':
    if "test" in sys.argv:
        del sys.argv[sys.argv.index("test")]
        unittest.main()

    I = DimSweep("I", [1, 2])
    J = DimSweep("J", [10, 20, 30])
    K = DimSweep("K", [100, 200, 300, 400])
    (i, j, k) = map(hfarray, (I, J, K))

    a = i * j * k
