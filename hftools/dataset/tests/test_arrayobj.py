#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os
import warnings
import numpy as np

from numpy import newaxis

import hftools.dataset.arrayobj as aobj
import hftools.dataset.dim as dim
import hftools.dataset as ds

from hftools.testing import TestCase, make_load_tests
from hftools.testing import random_value_array
from hftools.utils import reset_hftools_warnings, HFToolsDeprecationWarning

basepath = os.path.split(__file__)[0]
load_tests = make_load_tests(aobj)


class TestDims(TestCase):
    def test_contains1(self):
        dims = (dim.DimSweep("a", 3), dim.DimSweep("b", 3),
                dim.DimSweep("c", 3))
        dims = aobj.Dims(dims)
        for d in dims:
            self.assertTrue(d in dims)
#        self.assertFalse("a" in dims)

    def test_get_matching_dim1(self):
        dims = aobj.Dims((dim.DimSweep("a", 3), dim.DimSweep("b", 3),
                          dim.DimSweep("c", 3)))
        self.assertEqual(dims.get_matching_dim(dim.DimSweep("a", 4)), dims[0])
        self.assertEqual(dims.get_matching_dim(dim.DimSweep("b", 2)), dims[1])
        self.assertEqual(dims.get_matching_dim(dim.DimSweep("c", 4)), dims[2])

    def test_get_matching_dim_error1(self):
        dims = aobj.Dims((dim.DimSweep("a", 3), dim.DimSweep("b", 3),
                          dim.DimSweep("c", 3)))
        self.assertRaises(KeyError, dims.get_matching_dim, dim.DimRep("c", 4))
        self.assertRaises(KeyError, dims.get_matching_dim,
                          dim.DimSweep("d", 4))
        self.assertRaises(KeyError, dims.get_matching_dim, "a")

    def test_matching_index1(self):
        dims = aobj.Dims((dim.DimSweep("a", 3), dim.DimSweep("b", 3),
                          dim.DimSweep("c", 3)))
        self.assertEqual(dims.matching_index(dim.DimSweep("a", 4)), 0)
        self.assertEqual(dims.matching_index(dim.DimSweep("b", 2)), 1)
        self.assertEqual(dims.matching_index(dim.DimSweep("c", 4)), 2)

    def test_matching_index_error1(self):
        dims = aobj.Dims((dim.DimSweep("a", 3), dim.DimSweep("b", 3),
                          dim.DimSweep("c", 3)))
        self.assertRaises(KeyError, dims.matching_index, dim.DimRep("c", 4))
        self.assertRaises(KeyError, dims.matching_index, dim.DimSweep("d", 4))
        self.assertRaises(KeyError, dims.matching_index, "a")

    def test_containing_1(self):
        dims = aobj.Dims((dim.DimSweep("a", 3), dim.DimSweep("b", 3),
                          dim.DimSweep("c", 3)))
        self.assertTrue("c" in dims)
        self.assertFalse("d" in dims)


class TestDimsList(TestCase):
    def test_contains1(self):
        dims = (dim.DimSweep("a", 3), dim.DimSweep("b", 3),
                dim.DimSweep("c", 3))
        dims = aobj.DimsList(dims)
        for d in dims:
            self.assertTrue(d in dims)
        self.assertFalse("a" in dims)

    def test_get_matching_dim1(self):
        dims = aobj.DimsList((dim.DimSweep("a", 3), dim.DimSweep("b", 3),
                              dim.DimSweep("c", 3)))
        self.assertEqual(dims.get_matching_dim(dim.DimSweep("a", 4)), dims[0])
        self.assertEqual(dims.get_matching_dim(dim.DimSweep("b", 2)), dims[1])
        self.assertEqual(dims.get_matching_dim(dim.DimSweep("c", 4)), dims[2])

    def test_get_matching_dim_error1(self):
        dims = aobj.DimsList((dim.DimSweep("a", 3), dim.DimSweep("b", 3),
                              dim.DimSweep("c", 3)))
        self.assertRaises(KeyError, dims.get_matching_dim, dim.DimRep("c", 4))
        self.assertRaises(KeyError, dims.get_matching_dim,
                          dim.DimSweep("d", 4))
        self.assertRaises(KeyError, dims.get_matching_dim, "a")

    def test_matching_index1(self):
        dims = aobj.DimsList((dim.DimSweep("a", 3), dim.DimSweep("b", 3),
                              dim.DimSweep("c", 3)))
        self.assertEqual(dims.matching_index(dim.DimSweep("a", 4)), 0)
        self.assertEqual(dims.matching_index(dim.DimSweep("b", 2)), 1)
        self.assertEqual(dims.matching_index(dim.DimSweep("c", 4)), 2)

    def test_matching_index_error1(self):
        dims = aobj.DimsList((dim.DimSweep("a", 3), dim.DimSweep("b", 3),
                              dim.DimSweep("c", 3)))
        self.assertRaises(KeyError, dims.matching_index, dim.DimRep("c", 4))
        self.assertRaises(KeyError, dims.matching_index, dim.DimSweep("d", 4))
        self.assertRaises(KeyError, dims.matching_index, "a")


class Test_as_strided(TestCase):
    def test_1(self):
        a = np.arange(10)
        b = aobj.as_strided(a)
        self.assertAllclose(a, b)

    def test_2(self):
        a = np.arange(10)
        b = aobj.as_strided(a, (8,), a.strides, a.strides[0] * 2)
        self.assertAllclose(a[2:], b)

    def test_3(self):
        a = np.arange(10)
        b = aobj.as_strided(a, (4,), (a.strides[0] * 2,), a.strides[0] * 2)
        self.assertAllclose(a[2::2], b)

    def test_4(self):
        a = np.arange(10)
        b = aobj.as_strided(a, (8,))
        self.assertAllclose(a[:8], b)


class Test_expand_diagonals(TestCase):
    def setUp(self):
        self.a = aobj.hfarray([1, 2, 3], dims=(ds.DiagAxis("f", 3),))
        bdims = (ds.DiagAxis("f", 3), ds.DiagAxis("p", 3),)
        self.b = aobj.hfarray([[1, 2, 3],
                               [10, 20, 30],
                               [100, 200, 300]],
                              dims=bdims)

    def test_1(self):
        w = aobj.expand_diagonals(self.a)
        self.assertAllclose(w, np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))
        dims = (ds.IndepAxis("f", 3), ds.DerivAxis("f", 3),)
        self.assertEqual(w.dims, dims)

    def test_2(self):
        w = aobj.expand_diagonals(self.a, (ds.DiagAxis("f", 3),))
        self.assertAllclose(w, np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))
        dims = (ds.IndepAxis("f", 3), ds.DerivAxis("f", 3),)
        self.assertEqual(w.dims, dims)

    def test_3(self):
        w = aobj.expand_diagonals(self.b)
        B = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                B[i, i, j, j] = (j + 1) * 10 ** i
        self.assertAllclose(w, B)
        dims = (ds.IndepAxis("f", 3), ds.DerivAxis("f", 3),
                ds.IndepAxis("p", 3), ds.DerivAxis("p", 3),)
        self.assertEqual(w.dims, dims)

    def test_4(self):
        w = aobj.expand_diagonals(self.b, (ds.DiagAxis("f", 3),))
        B = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                B[i, i, j] = (j + 1) * 10 ** i
        self.assertAllclose(w, B)
        dims = (ds.IndepAxis("f", 3), ds.DerivAxis("f", 3),
                ds.DiagAxis("p", 3), )
        self.assertEqual(w.dims, dims)

    def test_5(self):
        w = aobj.expand_diagonals(self.b, (ds.DiagAxis("p", 3),))
        B = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                B[i, j, j] = (j + 1) * 10 ** i
        self.assertAllclose(w, B)
        dims = (ds.DiagAxis("f", 3), ds.IndepAxis("p", 3),
                ds.DerivAxis("p", 3), )
        self.assertEqual(w.dims, dims)

    def test_error_1(self):
        self.assertRaises(Exception, aobj.expand_diagonals,
                          self.b, (ds.IndepAxis("p", 3),))

    def test_error_2(self):
        self.assertRaises(Exception, aobj.expand_diagonals,
                          self.a, (ds.DiagAxis("p", 3),))


class Test_make_same_dims(TestCase):
    def setUp(self):
        self.a = aobj.hfarray([1, 2, 3], dims=(ds.DiagAxis("f", 3),))
        dims = (ds.DiagAxis("f", 3), ds.DiagAxis("p", 3),)
        self.b = aobj.hfarray([[1, 2, 3],
                               [10, 20, 30],
                               [100, 200, 300]], dims=dims)

    def test_1(self):
        a, b = aobj.make_same_dims(self.a, self.a)
        self.assertEqual(a.dims, b.dims)

    def test_2(self):
        a, b = aobj.make_same_dims(self.a, self.b)
        self.assertEqual(a.dims, b.dims)
        self.assertEqual(a.dims, (ds.DiagAxis("f", 3), ds.DiagAxis("p", 3),))

    def test_3(self):
        b, a = aobj.make_same_dims(self.b, self.a)
        self.assertEqual(a.dims, b.dims)
        self.assertEqual(a.dims, (ds.DiagAxis("f", 3), ds.DiagAxis("p", 3),))

    def test_4(self):
        a, b = aobj.make_same_dims(self.a, np.array(self.a))
        self.assertEqual(a.dims, b.dims)
        self.assertEqual(a.dims, (ds.DiagAxis("f", 3), ))


class Test_remove_tail(TestCase):
    def setUp(self):
        self.a = aobj.hfarray([1, 2, 3], dims=(ds.DiagAxis("f", 3),))
        dims = (ds.DiagAxis("f", 3), ds.DiagAxis("p", 3),)
        self.b = aobj.hfarray([[1, 2, 3],
                               [10, 20, 30],
                               [100, 200, 300]], dims=dims)
        dims = (ds.DimSweep("f", 2), ds.DimSweep("Vg", 3), ds.DimRep("rep", 4))
        self.c = aobj.hfarray(np.zeros((2, 3, 4)), dims=dims)
        dims = (ds.DimSweep("f", 2), ds.DimSweep("Vg", 3), ds.DimRep("rep", 4),
                ds.DimMatrix_i("i", 5), ds.DimMatrix_j("j", 5),)
        self.d = aobj.hfarray(np.zeros((2, 3, 4, 5, 5)), dims=dims)

    def test_1(self):
        a = aobj.remove_tail(self.a)
        self.assertAllclose(a, np.array(self.a)[:, newaxis])
        self.assertEqual(a.dims, (ds.DiagAxis("f", 3), ds.DimRep("Tail", 1)))

    def test_2(self):
        a = aobj.remove_tail(self.c)
        self.assertAllclose(a, 0)
        self.assertEqual(a.dims, (ds.DimSweep("f", 2), ds.DimRep("Tail", 12)))

    def test_3(self):
        a = aobj.remove_tail(self.d)
        self.assertAllclose(a, 0)
        dims = (ds.DimSweep("f", 2), ds.DimRep("Tail", 12),
                ds.DimMatrix_i("i", 5), ds.DimMatrix_j("j", 5),)
        self.assertEqual(a.dims, dims)

    def test_4(self):
        a = aobj.remove_tail(np.array(self.c))
        self.assertAllclose(a, 0)
        self.assertEqual(a.shape, (2, 12))

    def test_5(self):
        a = aobj.remove_tail(np.array(self.d))
        self.assertAllclose(a, 0)
        self.assertEqual(a.shape, (2, 12 * 25))


class Test_remove_rep(TestCase):
    def setUp(self):
        self.a = aobj.hfarray([1, 2, 3], dims=(ds.DiagAxis("f", 3),))
        dims = (ds.DiagAxis("f", 3), ds.DiagAxis("p", 3),)
        self.b = aobj.hfarray([[1, 2, 3],
                               [10, 20, 30],
                               [100, 200, 300]],
                              dims=dims)
        dims = (ds.DimSweep("f", 2), ds.DimSweep("Vg", 3), ds.DimRep("rep", 4))
        self.c = aobj.hfarray(np.zeros((2, 3, 4)), dims=dims)
        dims = (ds.DimSweep("f", 2), ds.DimRep("rep1", 3),
                ds.DimRep("rep2", 4), ds.DimMatrix_i("i", 5),
                ds.DimMatrix_j("j", 5),)
        self.d = aobj.hfarray(np.zeros((2, 3, 4, 5, 5)), dims=dims)

    def test_1(self):
        a = aobj.remove_rep(self.a)
        self.assertAllclose(a, self.a)
        self.assertEqual(a.dims, self.a.dims)

    def test_2(self):
        b = aobj.remove_rep(self.b)
        self.assertAllclose(b, self.b)
        self.assertEqual(b.dims, self.b.dims)

    def test_3(self):
        c = aobj.remove_rep(self.c)
        self.assertAllclose(c, 0)
        dims = (ds.DimSweep("f", 2), ds.DimSweep("Vg", 3),
                ds.DimRep("AllReps", 4))
        self.assertEqual(c.dims, dims)

    def test_4(self):
        d = aobj.remove_rep(self.d)
        self.assertAllclose(d, 0)
        dims = (ds.DimSweep("f", 2), ds.DimRep("AllReps", 12),
                ds.DimMatrix_i("i", 5), ds.DimMatrix_j("j", 5),)
        self.assertEqual(d.dims, dims)


class Test_make_complex_array(TestCase):
    def setUp(self):
        self.a = aobj.hfarray([1, 2, 3], dims=(ds.DiagAxis("f", 3),))
        dims = (ds.DiagAxis("f", 3),)
        self.ca = aobj.hfarray([1 + 10j, 2 + 20j, 3 + 30j], dims=dims)
        dims = (ds.DiagAxis("f", 3), ds.DiagAxis("p", 2),)
        self.b = aobj.hfarray([[1, 2],
                               [10, 20],
                               [100, 200]], dims=dims)

        self.d = aobj.hfarray([[1, -2], [2, 1]], dims=ds.CPLX)

    def test_1(self):
        a = aobj.make_fullcomplex_array(self.a)
        self.assertAllclose(a, np.array([[[1, 0],
                                          [0, 0]],
                                         [[2, 0],
                                          [0, 0]],
                                         [[3, 0],
                                          [0, 0]]]))
        self.assertEqual(a.dims, (ds.DiagAxis("f", 3),) + ds.CPLX)

    def test_2(self):
        ca = aobj.make_fullcomplex_array(self.ca)
        self.assertAllclose(ca, np.array([[[1, -10],
                                           [10, 1]],
                                          [[2, -20],
                                           [20, 2]],
                                          [[3, -30],
                                           [30, 3]]]))
        self.assertEqual(ca.dims, (ds.DiagAxis("f", 3),) + ds.CPLX)

    def test_3(self):
        d = aobj.make_fullcomplex_array(self.d)
        self.assertAllclose(d, self.d)
        self.assertEqual(d.dims, self.d.dims)

    def test_4(self):
        d = aobj.make_fullcomplex_array(np.array(1 + 1j))
        self.assertAllclose(d, np.array([[1, -1], [1, 1]]))
        self.assertEqual(d.dims, ds.CPLX)


class Test_change_shape(TestCase):
    def setUp(self):
        self.a = aobj.hfarray([1, 2, 3], dims=(ds.DiagAxis("f", 3),))
        dims = (ds.DiagAxis("f", 3),)
        self.ca = aobj.hfarray([1 + 10j, 2 + 20j, 3 + 30j], dims=dims)
        dims = (ds.DiagAxis("f", 3), ds.DiagAxis("p", 2),)
        self.b = aobj.hfarray([[1, 2], [10, 20], [100, 200]], dims=dims)

    def test_1(self):
        w = aobj.change_shape(self.a, (ds.DiagAxis("f", 3),))
        self.assertAllclose(w, self.a)
        self.assertEqual(w.dims, self.a.dims)

    def test_2(self):
        dims = (ds.DiagAxis("f", 3), ds.DiagAxis("p", 2),)
        w = aobj.change_shape(self.b, dims)
        self.assertAllclose(w, self.b)
        self.assertEqual(w.dims, self.b.dims)

    def test_3(self):
        dims = (ds.DiagAxis("p", 2), ds.DiagAxis("f", 3), )
        w = aobj.change_shape(self.b, dims)
        self.assertAllclose(w, self.b.T)
        self.assertEqual(w.dims, self.b.dims[::-1])

    def test_4(self):
        w = aobj.change_shape(self.a, (ds.DiagAxis("f", 3),) + ds.CPLX)
        self.assertAllclose(w, np.array([[[1, 0], [0, 0]],
                                         [[2, 0], [0, 0]],
                                         [[3, 0], [0, 0]]]))
        self.assertEqual(w.dims, (ds.DiagAxis("f", 3),) + ds.CPLX)

    def test_5(self):
        w = aobj.change_shape(self.ca, (ds.DiagAxis("f", 3),) + ds.CPLX)
        self.assertAllclose(w, np.array([[[1, -10], [10, 1]],
                                         [[2, -20], [20, 2]],
                                         [[3, -30], [30, 3]]]))
        self.assertEqual(w.dims, (ds.DiagAxis("f", 3),) + ds.CPLX)

    def test_error_1(self):
        dims = (ds.DiagAxis("f", 3), )
        self.assertRaises(ValueError, aobj.change_shape, self.b, dims)


class MakeData(object):
    def setUp(self):
        self.adims = (aobj.DimSweep("f", 1),
                      aobj.DimMatrix_i("i", 2),
                      aobj.DimMatrix_j("j", 2))
        self.a = aobj.hfarray([[[0, 0j], [0, 0]]], dims=self.adims)
        self.bdims = (aobj.DimSweep("f", 10),
                      aobj.DimMatrix_i("i", 2),
                      aobj.DimMatrix_j("j", 2))
        self.b = aobj.hfarray(np.array([[[1, 2 + 0j], [3, 4]]]) +
                              np.arange(10)[:, newaxis, newaxis],
                              dims=self.bdims)

        self.c, = aobj.make_same_dims_list([random_value_array(4, 5)])
        self.cdims = self.c.dims
        self.ddims = self.bdims
        self.d = aobj.hfarray(np.array([[[1, 2], [3, 4]]]) +
                              np.arange(10)[:, newaxis, newaxis],
                              dims=self.ddims)

        self.i1 = i1 = aobj.DimSweep("i1", 1)
        self.i2 = i2 = aobj.DimSweep("i2", 1)
        self.i3 = i3 = aobj.DimSweep("i3", 1)
        self.i4 = i4 = aobj.DimSweep("i4", 1)
        self.i5 = i5 = aobj.DimSweep("i5", 1)
        self.fi = fi = aobj.DimSweep("f", 2),
        self.gi = gi = aobj.DimSweep("g", 3),

        self.e = aobj.hfarray(np.zeros((1, 2, 3)), dims=(i1, fi, gi))
        self.f = aobj.hfarray(np.zeros((2, 3, 1)), dims=(fi, gi, i1))
        self.g = aobj.hfarray(np.zeros((1, 2, 3, 1)), dims=(i2, fi, gi, i1))
        dims = (i2, fi, i3, gi, i1)
        self.h = aobj.hfarray(np.zeros((1, 2, 1, 3, 1)), dims=dims)
        dims = (i4, i5)


class _Test_hfarray(MakeData, TestCase):
    pass


class Test_hfarray_checkinstance(TestCase):
    def setUp(self):
        class VArray(aobj._hfarray):
            __array_priority__ = 10
        self.VArray = VArray

    def test_1(self):
        a = self.VArray([1])
        b = aobj.hfarray([2])
        #None is placeholder for function that should never be called
        chk = aobj.check_instance(None)
        self.assertEqual(chk(b, a), NotImplemented)


class Test_hfarray_1(_Test_hfarray):

    def test_init_copy_1(self):
        b = aobj.hfarray(self.a)
        b[0] = 1
        self.assertAllclose(self.a, 0)
        self.assertAllclose(b, 1)

    def test_init_copy_2(self):
        b = aobj.hfarray(self.a, copy=False)
        b[0] = 1
        self.assertAllclose(self.a, 1)
        self.assertAllclose(b, 1)

    def test_copy_1(self):
        b = self.a.copy()
        b[0] = 1
        self.assertAllclose(self.a, 0)
        self.assertAllclose(b, 1)

    def test_init_1(self):
        a = np.array([1, 2, 3])
        A = aobj.hfarray(a)
        self.assertEqual(A.dims, (ds.DimSweep("freq", 3),))
        self.assertAllclose(a, A)

    def test_init_2(self):
        a = np.array([[1, 2, 3]] * 4)
        A = aobj.hfarray(a)
        self.assertEqual(A.dims, (ds.DimSweep("freq", 4), ds.DimRep("rep", 3)))
        self.assertAllclose(a, A)

    def test_init_3(self):
        a = aobj.hfarray(self.a, outputformat="%.3f")
        self.assertEqual(a.unit, self.a.unit)
        self.assertEqual(a.dims, self.a.dims)
        self.assertEqual(a.outputformat, "%.3f")
        self.assertAllclose(a, self.a)

    def test_init_4(self):
        fi = aobj.DimSweep("f", [12, 13, 14], unit="Hz", outputformat="%.3f")
        a = aobj.hfarray(fi)
        self.assertEqual(a.outputformat, "%.3f")
        self.assertEqual(a.unit, "Hz")
        self.assertEqual(a.dims, (fi,))
        self.assertAllclose(a, [12, 13, 14])

    def test_init_5(self):
        fi = aobj.DimSweep("f", [12, 13, 14], unit="m", outputformat="%.3f")
        a = aobj.hfarray(fi, unit="Hz")
        self.assertEqual(a.unit, "Hz")
        self.assertEqual(a.outputformat, "%.3f")
        self.assertEqual(a.dims, (fi,))
        self.assertAllclose(a, [12, 13, 14])

    def test_init_6(self):
        fi = aobj.DimSweep("f", [12, 13, 14], unit="m", outputformat="%.3f")
        a = aobj.hfarray(fi, unit="Hz", outputformat="%.5f")
        self.assertEqual(a.unit, "Hz")
        self.assertEqual(a.dims, (fi,))
        self.assertEqual(a.outputformat, "%.5f")
        self.assertAllclose(a, [12, 13, 14])

    def test_init_error_1(self):
        a = np.array([[[1, 2, 3]] * 3] * 2)
        self.assertRaises(aobj.DimensionMismatchError, aobj.hfarray, a)

    def test_indexing_1(self):
        self.assertAllclose(self.b[0], np.array([[1, 2 + 0j], [3, 4]]))

    def test_indexing_2(self):
        facit = (np.array([[[1, 2 + 0j], [3, 4]]]) +
                 np.arange(2)[:, newaxis, newaxis])
        self.assertAllclose(self.b[:2], facit)

    def test_indexing_3(self):
        facit = (np.array([[[1, 2 + 0j], [3, 4]]]) +
                 np.arange(0, 10, 2)[:, newaxis, newaxis])
        self.assertAllclose(self.b[::2], facit)

    def test_verify_dimension_1(self):
        self.assertIsNone(self.a.verify_dimension())

    def test_verify_dimension_2(self):
        self.assertIsNone(self.b.verify_dimension())

    def test_verify_dimension_error_1(self):
        self.a.dims = self.a.dims[:-1]
        self.assertRaises(aobj.HFArrayShapeDimsMismatchError,
                          self.a.verify_dimension)

    def test_info_deprecation(self):
        a = aobj.hfarray(1)
        reset_hftools_warnings()
        self.assertHFToolsDeprecationWarning(lambda x: x.info, a)
        with warnings.catch_warnings():
            warnings.resetwarnings()
            warnings.simplefilter("ignore", HFToolsDeprecationWarning)
            a.info
##        reset_hftools_warnings()

    def test_dims_index_1(self):
        self.assertEqual(self.a.dims_index("f"), 0)
        self.assertEqual(self.a.dims_index("i"), 1)
        self.assertEqual(self.a.dims_index("j"), 2)
        self.assertEqual(self.a.dims_index("f", aobj.DimSweep), 0)
        self.assertEqual(self.a.dims_index("i", aobj.DimMatrix_i), 1)
        self.assertEqual(self.a.dims_index("j", aobj.DimMatrix_j), 2)

    def test_dims_index_2(self):
        self.assertEqual(self.b.dims_index("f"), 0)
        self.assertEqual(self.b.dims_index("i"), 1)
        self.assertEqual(self.b.dims_index("j"), 2)
        self.assertEqual(self.a.dims_index("f", aobj.DimSweep), 0)
        self.assertEqual(self.a.dims_index("i", aobj.DimMatrix_i), 1)
        self.assertEqual(self.a.dims_index("j", aobj.DimMatrix_j), 2)

    def test_dims_index_error_1(self):
        self.assertRaises(IndexError, self.a.dims_index, "Q")
        self.assertRaises(IndexError, self.a.dims_index, "f", aobj.DimMatrix_i)
        self.assertRaises(IndexError, self.a.dims_index, "i", aobj.DimSweep)
        self.assertRaises(IndexError, self.a.dims_index, "j", aobj.DimSweep)

    def test_dims_index_error_2(self):
        self.assertRaises(IndexError, self.b.dims_index, "X")
        self.assertRaises(IndexError, self.b.dims_index, "f", aobj.DimMatrix_i)
        self.assertRaises(IndexError, self.b.dims_index, "i", aobj.DimSweep)
        self.assertRaises(IndexError, self.b.dims_index, "j", aobj.DimSweep)

    def test_help_1(self):
        self.a.help()

    def test_help_2(self):
        self.b.help()

    def test_reorder_1(self):
        dims = self.c.dims
        a = self.c.reorder_dimensions(*dims[-1:])
        self.assertEqual(a.dims, dims[-1:] + dims[:-1])

    def test_reorder_2(self):
        dims = self.c.dims
        a = self.c.reorder_dimensions(*dims[-2:])
        self.assertEqual(a.dims, dims[-2:] + dims[:-2])

    def test_reorder_3(self):
        dims = self.c.dims
        a = self.c.reorder_dimensions(*dims[::-1])
        self.assertEqual(a.dims, dims[::-1])

    def test_squeeze_1(self):
        a = self.e.squeeze()
        self.assertEqual(a.dims, (self.fi, self.gi))

    def test_squeeze_2(self):
        a = self.f.squeeze()
        self.assertEqual(a.dims, (self.fi, self.gi))

    def test_squeeze_3(self):
        a = self.g.squeeze()
        self.assertEqual(a.dims, (self.fi, self.gi))

    def test_squeeze_4(self):
        a = self.h.squeeze()
        self.assertEqual(a.dims, (self.fi, self.gi))


class Test_hfarray_getitem(_Test_hfarray):
    def test_getitem_1(self):
        q = self.a[...]
        self.assertEqual(q.__array_interface__, self.a.__array_interface__)


class Test_rss_method(_Test_hfarray):
    def test_1(self):
        v, = aobj.make_same_dims_list([random_value_array(4, 5)])
        a = np.array(v)
        facit = np.sqrt((abs(a) ** 2).sum())
        res = v.rss()
        self.assertAllclose(res, facit)

    def test_2(self):
        v, = aobj.make_same_dims_list([random_value_array(4, 5)])
        a = np.array(v)
        facit = np.sqrt((abs(a) ** 2).sum(0))
        res = v.rss(0)
        self.assertAllclose(res, facit)

    def test_3(self):
        v, = aobj.make_same_dims_list([random_value_array(4, 5)])
        a = np.array(v)
        facit = np.sqrt((abs(a) ** 2).sum(1))
        res = v.rss(1)
        self.assertAllclose(res, facit)


class Test_cumsum(_Test_hfarray):
    methodname = "cumsum"
    kw = {}

    def test_1(self):
        v = random_value_array(4, 5, minsize=2)
        a = np.array(v)

        args = self.kw.copy()
        for i in range(v.ndim):
            r1 = getattr(v, self.methodname)(axis=i, **args)
            r2 = getattr(a, self.methodname)(axis=i, **args)
            r1.verify_dimension()
            self.assertTrue(np.allclose(r1, r2))
            self.assertIsInstance(r1, v.__class__)

    def test_index_error(self):
        v = random_value_array(4, 5, minsize=2)
        method = getattr(v, self.methodname)
        self.assertRaises(IndexError, method, "NONEXISTING")


class Test_mean(Test_cumsum):
    methodname = "mean"

    def test_axis_none(self):
        v = random_value_array(4, 5)
        a = np.array(v)
        args = self.kw.copy()
        args.update(axis=None)
        r1 = getattr(v, self.methodname)(**args)
        r2 = getattr(a, self.methodname)(**args)
        r1.verify_dimension()
        self.assertTrue(np.allclose(r1, r2))
        self.assertIsInstance(r1, v.__class__)

    def test_axis_dimsweep_class(self):
        v = self.b
        a = np.array(v)
        args = self.kw.copy()
        r1 = getattr(v, self.methodname)(axis=aobj.DimSweep, **args)
        r2 = getattr(a, self.methodname)(axis=0, **args)
        r1.verify_dimension()
        self.assertTrue(np.allclose(r1, r2))
        self.assertIsInstance(r1, v.__class__)

    def test_defaults(self):
        v = random_value_array(4, 5)
        a = np.array(v)
        r1 = getattr(v, self.methodname)()
        r2 = getattr(a, self.methodname)()
        r1.verify_dimension()
        self.assertTrue(np.allclose(r1, r2))
        self.assertIsInstance(r1, v.__class__)

    def test_axis_specified_1(self):
        v = self.b
        a = np.array(v)
        args = self.kw.copy()
        r1 = getattr(v, self.methodname)(axis=self.bdims[0], **args)
        r2 = getattr(a, self.methodname)(axis=0, **args)
        r1.verify_dimension()
        self.assertTrue(np.allclose(r1, r2))
        self.assertIsInstance(r1, v.__class__)

    def test_axis_specified_2(self):
        v = self.b
        a = np.array(v)
        args = self.kw.copy()
        r1 = getattr(v, self.methodname)(axis=self.bdims[:1], **args)
        r2 = getattr(a, self.methodname)(axis=0, **args)
        r1.verify_dimension()
        self.assertTrue(np.allclose(r1, r2))
        self.assertIsInstance(r1, v.__class__)

    def test_axis_specified_not_in_dims(self):
        self.assertRaises(IndexError, getattr(self.a, self.methodname),
                          aobj.DimRep("rep", 4))

    def test_axis_specified_not_in_dims_2(self):
        res = self.a.mean(axis=aobj.DimRep("rep", 4), dimerror=False)
        self.assertAllclose(res, self.a)

    def test_index_error_dont_raise(self):
        v = random_value_array(4, 5, minsize=2)
        method = getattr(v, self.methodname)
        q = method("NONEXISTING", dimerror=False)
        self.assertAllclose(q, v)

    def test_keepdims(self):
        v = random_value_array(4, 5, minsize=2)
        a = np.array(v)

        args = self.kw.copy()
        args["keepdims"] = True
        for i in range(v.ndim):
            r1 = getattr(v, self.methodname)(axis=i, **args)
            r2 = getattr(a, self.methodname)(axis=i, **args)
            r1.verify_dimension()
            self.assertTrue(np.allclose(r1, r2))
            self.assertIsInstance(r1, v.__class__)


class Test_std(Test_mean):
    methodname = "std"


class Test_var(Test_mean):
    methodname = "var"


class Test_max(Test_mean):
    methodname = "max"


class Test_min(Test_mean):
    methodname = "min"


class Test_sum(Test_mean):
    methodname = "sum"


class Test_cumsum2(_Test_hfarray):
    methodname = "cumsum"

    def test_error_1(self):
        v = random_value_array(4, 5)
        self.assertRaises(aobj.HFArrayError, getattr(v, self.methodname),
                          None)

    def test_error_2(self):
        v = random_value_array(4, 5)
        self.assertRaises(IndexError, getattr(v, self.methodname), (0, 1))


class Test_cumprod(Test_cumsum):
    methodname = "cumprod"


class Test_cumprod2(Test_cumsum2):
    methodname = "cumprod"


class Test_make_matrix(TestCase):
    def test_1(self):
        a = random_value_array(3, 5)
        m = aobj.make_matrix(np.array(a), a.dims[:-2])
        self.assertAllclose(m, a)
        self.assertEqual(m.dims[-2:], (ds.DimMatrix_i("i", a.shape[-2]),
                                       ds.DimMatrix_j("j", a.shape[-1])))

    def test_2(self):
        a = random_value_array(3, 5)
        self.assertRaises(aobj.HFArrayShapeDimsMismatchError,
                          aobj.make_matrix, np.array(a), a.dims[:2])

    def test_3(self):
        a = random_value_array(3, 5)
        self.assertRaises(aobj.HFArrayShapeDimsMismatchError,
                          aobj.make_matrix, np.array(a), a.dims[:3])


class Test_make_vector(TestCase):
    def test_1(self):
        a = random_value_array(3, 5)
        m = aobj.make_vector(np.array(a), a.dims[:-1])
        self.assertAllclose(m, a)
        self.assertEqual(m.dims[-1:], (ds.DimMatrix_j("j", a.shape[-1]),))

    def test_2(self):
        a = random_value_array(3, 5)
        self.assertRaises(aobj.HFArrayShapeDimsMismatchError,
                          aobj.make_vector, np.array(a), a.dims[:])

    def test_3(self):
        a = random_value_array(3, 5)
        self.assertRaises(aobj.HFArrayShapeDimsMismatchError,
                          aobj.make_vector, np.array(a), a.dims[:1])

    def test_4(self):
        a = random_value_array(3, 5)
        self.assertRaises(aobj.HFArrayShapeDimsMismatchError,
                          aobj.make_vector, np.array(a), a.dims[:0])


class Test_multiple_axis_handler(TestCase):
    def test1(self):
        self.assertRaises(IndexError,
                          aobj.multiple_axis_handler, aobj.hfarray(1),
                          "APA")


class Test_replace_dim(TestCase):
    def test_1(self):
        ai = aobj.DimSweep("a", [1, 2, 3])
        bi = aobj.DimSweep("b", [1, 2])
        A = aobj.hfarray(ai)
        B = aobj.hfarray(bi)
        AB = A * B
        newdim = aobj.DimSweep("c", [1, 2, 3])
        AB.replace_dim(ai, newdim)
        self.assertEqual(AB.dims, (newdim, bi))
        AB = A * B
        AB.replace_dim("a", newdim)
        self.assertEqual(AB.dims, (newdim, bi))

    def test_2(self):
        ai = aobj.DimSweep("a", [1, 2, 3])
        bi = aobj.DimSweep("b", [1, 2])
        aj = aobj.DimRep("a", [1, 2, 3])
        A = aobj.hfarray(ai)
        B = aobj.hfarray(bi)
        AB = A * B
        AB.replace_dim("a", aobj.DimRep)
        self.assertEqual(AB.dims, (aj, bi))


if __name__ == '__main__':
    def test_1(methodname="argsort", **kw):
        v = random_value_array(4, 5)
        a = np.array(v)
        for i in range(v.ndim):
            r1 = getattr(a, methodname)(axis=i, **kw)
            r2 = getattr(v, methodname)(axis=i, **kw)
            print i, np.allclose(r1, r2), isinstance(r2, v.__class__)
        return v, a
