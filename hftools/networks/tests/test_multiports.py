# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os
import pdb
import unittest2 as unittest

import numpy as np

import hftools.dataset.arrayobj as aobj
import hftools.networks.multiports as mp
from hftools.testing import TestCase, random_complex_value_array,\
    random_complex_matrix

basepath = os.path.split(__file__)[0]

network_classes = [v for k, v in mp.__dict__.items()
                   if (isinstance(v, type) and
                       issubclass(v, mp._MultiPortArray) and
                       v is not mp._MultiPortArray and
                       v is not mp._TwoPortArray)]


class _Test_hfarray(TestCase):
    def setUp(self):
        adims = (aobj.DimSweep("f", 1),
                 aobj.DimMatrix_i("i", 2),
                 aobj.DimMatrix_j("j", 2))
        self.a = aobj.hfarray([[[1, 2 + 0j], [3, 4]]], dims=adims)
        bdims = (aobj.DimSweep("f", 1),
                 aobj.DimMatrix_i("i", 3),
                 aobj.DimMatrix_j("j", 3))
        self.b = aobj.hfarray([[[1, 2 + 0j, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]]], dims=bdims)


class Test_TwoPortArray(_Test_hfarray):
    cls = mp._TwoPortArray

    def test_must_be_twoport(self):
        self.assertRaises(ValueError, self.cls, self.b)

    def test_init_copy_1(self):
        b = self.cls(self.a)
        b[0] = 1
        self.assertTrue(np.allclose(np.array(self.a),
                        np.array([[1, 2], [3, 4]])))
        self.assertTrue(np.allclose(b, 1))

    def test_init_copy_2(self):
        b = self.cls(self.a, copy=False)
        b[0] = 1
        self.assertTrue(np.allclose(self.a, 1))
        self.assertTrue(np.allclose(b, 1))

    def test_via_shortname_1(self):
        if self.cls.shortname:
            b = self.cls(self.a, copy=False)
            self.assertAllclose(b[..., 0, 0],
                                getattr(b, "%s11" % b.shortname))
            self.assertAllclose(b[..., 0, 1],
                                getattr(b, "%s12" % b.shortname))
            self.assertAllclose(b[..., 1, 0],
                                getattr(b, "%s21" % b.shortname))
            self.assertAllclose(b[..., 1, 1],
                                getattr(b, "%s22" % b.shortname))

    def test_cascade(self):
        if self.cls == mp._TwoPortArray:
            return
        A = self.cls(mp.SArray([[0.1, 0.2], [0.3, 0.4]]))
        B = self.cls(mp.SArray([[0.4, 0.3], [0.2, 0.1]]))
        Q = A.cascade(B)
        facit = [[0.1285714285714286, 0.07142857142857142],
                 [0.07142857142857142, 0.1285714285714286]]
        self.assertIsInstance(Q, self.cls)
        self.assertAllclose(mp.SArray(Q), facit)

    def test_deembed_none(self):
        if self.cls == mp._TwoPortArray:
            return
        A = self.cls(mp.SArray([[0.1, 0.2], [0.3, 0.4]]))
        Q = A.deembed()
        self.assertIsInstance(Q, self.cls)
        self.assertAllclose(mp.SArray(Q), [[0.1, 0.2], [0.3, 0.4]])

    def test_deembed_left(self):
        if self.cls == mp._TwoPortArray:
            return
        A = self.cls(mp.SArray([[0.1, 0.2], [0.3, 0.4]]))
        B = self.cls(mp.SArray([[0.4, 0.3], [0.2, 0.1]]))
        Q = A.deembed(left=B)
        facit = [[-10, 4 / 3.],
                 [3, 0.2]]
        self.assertIsInstance(Q, self.cls)
        self.assertAllclose(mp.SArray(Q), facit)

    def test_deembed_right(self):
        if self.cls == mp._TwoPortArray:
            return
        A = self.cls(mp.SArray([[0.1, 0.2], [0.3, 0.4]]))
        B = self.cls(mp.SArray([[0.4, 0.3], [0.2, 0.1]]))
        Q = A.deembed(right=B)
        facit = [[-1 / 30., 2 / 9.],
                 [.5, 5 / 3.]]
        self.assertIsInstance(Q, self.cls)
        self.assertAllclose(mp.SArray(Q), facit)

    def test_deembed_both(self):
        if self.cls == mp._TwoPortArray:
            return
        A = self.cls(mp.SArray([[0.1, 0.2], [0.3, 0.4]]))
        B = self.cls(mp.SArray([[0.4, 0.3], [0.2, 0.1]]))
        Q = A.deembed(right=B, left=B)
        facit = [[-26., 8 / 3.],
                 [9, 1.]]
        self.assertIsInstance(Q, self.cls)
        self.assertAllclose(mp.SArray(Q), facit)

    def test_1(self):
        res = self.cls(np.array(self.a))
        resdims = (aobj.DimSweep("freq", 1), ) + self.a.dims[1:]
        self.assertAllclose(res, self.a)
        self.assertEqual(res.dims, resdims)

    def test_2(self):
        self.assertRaises(aobj.DimensionMismatchError,
                          self.cls, np.array([[self.a]]))


class Test_MultiPortArray(_Test_hfarray):
    cls = mp._MultiPortArray

    def test_init_copy_1(self):
        b = self.cls(self.a)
        b[0] = 1
        self.assertTrue(np.allclose(np.array(self.a),
                        np.array([[1, 2], [3, 4]])))
        self.assertTrue(np.allclose(b, 1))

    def test_init_copy_2(self):
        b = self.cls(self.a, copy=False)
        b[0] = 1
        self.assertTrue(np.allclose(self.a, 1))
        self.assertTrue(np.allclose(b, 1))

    def test_via_shortname_1(self):
        if self.cls.shortname:
            b = self.cls(self.a, copy=False)
            self.assertAllclose(b[..., 0, 0],
                                getattr(b, "%s11" % b.shortname))
            self.assertAllclose(b[..., 0, 1],
                                getattr(b, "%s12" % b.shortname))
            self.assertAllclose(b[..., 1, 0],
                                getattr(b, "%s21" % b.shortname))
            self.assertAllclose(b[..., 1, 1],
                                getattr(b, "%s22" % b.shortname))

    def test_init_error(self):
        if self.cls == mp._MultiPortArray:
            return
        r = self.cls(self.b)[..., :2]
        for cls in [mp.SArray, mp.ZArray, mp.YArray]:
            if cls != self.cls:
                self.assertRaises(ValueError, cls, r)

    def test_cascade(self):
        if self.cls == mp._MultiPortArray:
            return
        A = self.cls(mp.SArray([[0.1, 0.2], [0.3, 0.4]]))
        B = self.cls(mp.SArray([[0.4, 0.3], [0.2, 0.1]]))
        Q = A.cascade(B)
        facit = [[0.1285714285714286, 0.07142857142857142],
                 [0.07142857142857142, 0.1285714285714286]]
        self.assertIsInstance(Q, self.cls)
        self.assertAllclose(mp.SArray(Q), facit)

    def test_deembed_none(self):
        if self.cls == mp._MultiPortArray:
            return
        A = self.cls(mp.SArray([[0.1, 0.2], [0.3, 0.4]]))
        Q = A.deembed()
        self.assertIsInstance(Q, self.cls)
        self.assertAllclose(mp.SArray(Q), [[0.1, 0.2], [0.3, 0.4]])

    def test_deembed_left(self):
        if self.cls == mp._MultiPortArray:
            return
        A = self.cls(mp.SArray([[0.1, 0.2], [0.3, 0.4]]))
        B = self.cls(mp.SArray([[0.4, 0.3], [0.2, 0.1]]))
        Q = A.deembed(left=B)
        facit = [[-10, 4 / 3.],
                 [3, 0.2]]
        self.assertIsInstance(Q, self.cls)
        self.assertAllclose(mp.SArray(Q), facit)

    def test_deembed_right(self):
        if self.cls == mp._MultiPortArray:
            return
        A = self.cls(mp.SArray([[0.1, 0.2], [0.3, 0.4]]))
        B = self.cls(mp.SArray([[0.4, 0.3], [0.2, 0.1]]))
        Q = A.deembed(right=B)
        facit = [[-1 / 30., 2 / 9.],
                 [.5, 5 / 3.]]
        self.assertIsInstance(Q, self.cls)
        self.assertAllclose(mp.SArray(Q), facit)

    def test_deembed_both(self):
        if self.cls == mp._MultiPortArray:
            return
        A = self.cls(mp.SArray([[0.1, 0.2], [0.3, 0.4]]))
        B = self.cls(mp.SArray([[0.4, 0.3], [0.2, 0.1]]))
        Q = A.deembed(right=B, left=B)
        facit = [[-26., 8 / 3.],
                 [9, 1.]]
        self.assertIsInstance(Q, self.cls)
        self.assertAllclose(mp.SArray(Q), facit)


class Test_ZArray_1(Test_MultiPortArray):
    cls = mp.ZArray

    def test_setattr_1(self):
        res = mp.ZArray(np.array(self.a))
        res.Z11 = 1.98765
        self.assertAllclose(res.Z11, 1.98765)
        self.assertAllclose(res[..., 0, 0], 1.98765)

    def test_setattr_2(self):
        res = mp.ZArray(np.array(self.a))
        res.__dict__["FOO"] = 10
        res.FOO = 20
        self.assertEqual(res.__dict__["FOO"], 20)


class Test_YArray_1(Test_MultiPortArray):
    cls = mp.YArray


class Test_GArray_1(Test_TwoPortArray):
    cls = mp.GArray


class Test_HArray_1(Test_TwoPortArray):
    cls = mp.HArray


class Test_ABCDArray_1(Test_TwoPortArray):
    cls = mp.ABCDArray


class Test_SArray_1(Test_MultiPortArray):
    cls = mp.SArray


class Test_TArray_1(Test_TwoPortArray):
    cls = mp.TArray


class Test_TPArray_1(Test_TwoPortArray):
    cls = mp.TpArray


class Test_unit_matrices(TestCase):
    def test_unit_matrix(self):
        m = mp.unit_matrix()
        self.assertEqual(m.shape, (2, 2))
        self.assertTrue(isinstance(m, mp.SArray))
        self.assertAllclose(m, [[1, 0], [0, 1]])

    def test_unit_smatrix(self):
        m = mp.unit_smatrix()
        self.assertEqual(m.shape, (2, 2))
        self.assertTrue(isinstance(m, mp.SArray))
        self.assertAllclose(m, [[0, 1], [1, 0]])


class TestConversions(TestCase):
    """test conversions by generating a random matrix of type startcls
       and then for all networkclasses (stopclass) convert from startcls
       to stopclass and back to startcls which should yield the same result.
    """
    def setUp(self):
        self.a = random_complex_matrix(3, 4, 1, 2)

    def _conversion(self, startcls):
        start = startcls(self.a)
        for c in network_classes:
            res = startcls(c(start))
            msg = "Conversion between %r and %r failed" % (startcls, c)
            self.assertAllclose(start, res, msg=msg)
            self.assertEqual(res.__class__, startcls)

    def testS(self):
        self._conversion(mp.SArray)

    def testT(self):
        self._conversion(mp.TArray)

    def testTp(self):
        self._conversion(mp.TpArray)

    def testZ(self):
        self._conversion(mp.ZArray)

    def testY(self):
        self._conversion(mp.YArray)

    def testG(self):
        self._conversion(mp.GArray)

    def testH(self):
        self._conversion(mp.HArray)


if __name__ == '__main__':
    unittest.main()
