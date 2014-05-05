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

import hftools.dataset.dim as ddim

from hftools.dataset.dim import DimSweep, ComplexDiagAxis, ComplexIndepAxis,\
    ComplexDerivAxis, dims_has_complex, info_has_complex
from hftools.testing import TestCase, make_load_tests
from hftools.dataset import hfarray
from hftools.utils import reset_hftools_warnings, HFToolsDeprecationWarning

basepath = os.path.split(__file__)[0]
load_tests = make_load_tests(ddim)


class Test_Dim_init(TestCase):
    def setUp(self):
        self.dim = DimSweep("a", 10, unit="Hz")

    def test_1(self):
        a = DimSweep(self.dim)
        self.assertEqual(a, self.dim)
        self.assertEqual(a.name, "a")
        self.assertEqual(a.unit, "Hz")
        self.assertAllclose(a.data, range(10))

    def test_2(self):
        a = DimSweep(self.dim, data=range(5))
        self.assertEqual(a.name, "a")
        self.assertEqual(a.unit, "Hz")
        self.assertAllclose(a.data, range(5))

    def test_3(self):
        a = DimSweep(self.dim, unit="m")
        self.assertEqual(a.name, "a")
        self.assertEqual(a.unit, "m")
        self.assertAllclose(a.data, range(10))

    def test_4(self):
        a = DimSweep(self.dim, name="P")
        self.assertEqual(a.name, "P")
        self.assertEqual(a.unit, "Hz")
        self.assertAllclose(a.data, range(10))

    def test_5(self):
        a = DimSweep(self.dim, name="P", unit="W", data=range(3))
        self.assertEqual(a.name, "P")
        self.assertEqual(a.unit, "W")
        self.assertAllclose(a.data, range(3))

    def test_6(self):
        a = DimSweep("P", data=set([0, 1, 2]),  unit="W")
        self.assertEqual(a.name, "P")
        self.assertEqual(a.unit, "W")
        self.assertAllclose(sorted(a.data), range(3))

    def test_7(self):
        a = DimSweep("P", data=np.array([0, 1, 2]),  unit="W")
        self.assertEqual(a.name, "P")
        self.assertEqual(a.unit, "W")
        self.assertAllclose(a.data, range(3))

    def test_hfarray(self):
        a = hfarray(self.dim)
        self.assertAllclose(a, range(10))
        self.assertEqual(a.dims, (DimSweep("a", 10, unit="Hz"),))


class Test_Dim(TestCase):
    cls = ddim.DimBase

    def _helper(self, name, indata, finaldata):
        d = self.cls("a", indata)
        self.assertEqual(d.name, "a")
        self.assertTrue(np.allclose(d.data, np.array(finaldata)))
        self.assertEqual(d.fullsize(), len(finaldata))

    def test_1(self):
        self._helper("a", 1, [0])

    def test_2(self):
        self._helper("a", 3, [0, 1, 2])

    def test_3(self):
        self._helper("a", [1, 2, 3], [1, 2, 3])

    def test_4(self):
        self._helper("a", [[1, 2, 3]], [1, 2, 3])

    def test_cmp_1(self):
        a = self.cls("a", 0)
        b = self.cls("b", 4)
        self.assertTrue(a < b)
        self.assertFalse(a > b)

    def test_cmp_2(self):
        a = self.cls("a", 4)
        b = self.cls("a", 4)
        self.assertEqual(a, b)

    def test_cmp_3(self):
        a = self.cls("a", 4)
        b = "a"
        self.assertEqual(a, b)

    def test_cmp_4(self):
        a = self.cls("a", 4)
        b = "b"
        res = a < b
        self.assertTrue(res)

    def test_slice_1(self):
        a = self.cls("a", 10)
        b = a[::2]
        self.assertEqual(b.name, a.name)
        self.assertTrue(np.allclose(a.data, np.arange(10)))
        self.assertTrue(np.allclose(b.data, np.arange(10)[::2]))
        self.assertEqual(a.fullsize(), 10)
        self.assertEqual(b.fullsize(), 5)

    def test_data(self):
        a = self.cls("a", 2)
        self.assertRaises(AttributeError, setattr, a, "data", "b")

    def test_name(self):
        a = self.cls("a", 2)
        self.assertRaises(AttributeError, setattr, a, "name", "b")

    def test_unit(self):
        a = self.cls("a", 2)
        self.assertRaises(AttributeError, setattr, a, "unit", "b")

    def test_hash(self):
        a = self.cls("a", 2)
        hashres = hash((self.cls.sortprio, "a", self.cls, (0, 1), None, None))
        self.assertEqual(hash(a), hashres)

    def test_get_1(self):
        a = self.cls("a", 10)
        self.assertEqual(a[:], a)

    def test_get_2(self):
        a = self.cls("a", 10, unit="Hz")
        b = a[::2]
        self.assertEqual(b.data.tolist(), a.data.tolist()[::2])
        self.assertEqual(b.unit, a.unit)
        self.assertEqual(b.name, a.name)

    def test_get_3(self):
        a = self.cls("a", 10, unit="Hz")
        b = a[np.arange(10) < 5]
        self.assertEqual(b.data.tolist(), list(range(5)))
        self.assertEqual(b.unit, a.unit)
        self.assertEqual(b.name, a.name)

    def test_get_4(self):
        a = self.cls("a", 10, unit="Hz")
        self.assertRaises(IndexError, lambda x: x[0], a)


class Test_DimSweep(Test_Dim):
    cls = ddim.DimSweep


class Test_DimRep(Test_Dim):
    cls = ddim.DimRep


class Test_DimMatrix(Test_Dim):
    cls = ddim._DimMatrix


class Test_DimMatrix_i(Test_Dim):
    cls = ddim.DimMatrix_i


class Test_DimMatrix_j(Test_Dim):
    cls = ddim.DimMatrix_j


class TestDiag(Test_Dim):
    cls = ddim.DiagAxis
    indep = ddim.IndepAxis
    deriv = ddim.DerivAxis

    def test_indep_1(self):
        a = self.cls("a", 10)
        self.assertIsInstance(a.indep_axis, self.indep)
        self.assertIsInstance(a.indep_axis.diag_axis, self.cls)
        self.assertIsInstance(a.indep_axis.deriv_axis, self.deriv)
        self.assertEqual(a.indep_axis.name, a.name)
        self.assertEqual(a.indep_axis.unit, a.unit)
        self.assertAllclose(a.indep_axis.data, a.data)

    def test_indep_2(self):
        a = self.cls("a", 10)
        self.assertIsInstance(a.deriv_axis, self.deriv)
        self.assertIsInstance(a.deriv_axis.diag_axis, self.cls)
        self.assertIsInstance(a.deriv_axis.indep_axis, self.indep)
        self.assertEqual(a.deriv_axis.name, a.name)
        self.assertEqual(a.deriv_axis.unit, a.unit)
        self.assertAllclose(a.deriv_axis.data, a.data)


class TestDiag_Complex(Test_Dim):
    cls = ddim.ComplexDiagAxis
    indep = ddim.ComplexIndepAxis
    deriv = ddim.ComplexDerivAxis


class TestDiag_Matrix_i(Test_Dim):
    cls = ddim.DimMatrix_i
    indep = ddim.DimMatrix_Indep_i
    deriv = ddim.DimMatrix_Deriv_i


class TestDiag_Matrix_j(Test_Dim):
    cls = ddim.DimMatrix_j
    indep = ddim.DimMatrix_Indep_j
    deriv = ddim.DimMatrix_Deriv_j


class Test_dims_has_complex(TestCase):
    def _helper(self, dims):
        self.assertTrue(dims_has_complex(dims))

    def _helper_false(self, dims):
        self.assertFalse(dims_has_complex(dims))

    def test_1(self):
        self._helper_false((DimSweep("d", 3),))

    def test_2(self):
        self._helper((DimSweep("d", 3), ComplexDiagAxis("cplx", 2)))

    def test_3(self):
        self._helper((DimSweep("d", 3), ComplexIndepAxis("cplx", 2)))

    def test_4(self):
        self._helper((DimSweep("d", 3), ComplexDerivAxis("cplx", 2)))

    def test_5(self):
        self._helper((DimSweep("d", 3), ComplexIndepAxis("cplx", 2),
                      ComplexDerivAxis("cplx", 2)))


class Test_info_has_complex(Test_dims_has_complex):
    def _helper(self, dims):
        reset_hftools_warnings()
        self.assertHFToolsDeprecationWarning(info_has_complex, dims)
        with warnings.catch_warnings():
            warnings.resetwarnings()
            warnings.simplefilter("ignore", HFToolsDeprecationWarning)
            self.assertTrue(info_has_complex(dims))

    def _helper_false(self, dims):
        reset_hftools_warnings()
        self.assertHFToolsDeprecationWarning(info_has_complex, dims)
        with warnings.catch_warnings():
            warnings.resetwarnings()
            warnings.simplefilter("ignore", HFToolsDeprecationWarning)
            self.assertFalse(info_has_complex(dims))


class TestDimConv(TestCase):
    a = ddim.DimBase
    b = ddim.DimBase

    def setUp(self):
        self.A = self.a("A", [0], unit="Hz")
        self.B = self.b("B", [0], unit="N")

    def test_self_conv_1(self):
        OBJ = self.a(self.A)
        self.assertEqual(OBJ.name, self.A.name)
        self.assertEqual(OBJ.data, self.A.data)
        self.assertEqual(OBJ.unit, self.A.unit)
        self.assertIsInstance(OBJ, self.a)

    def test_self_conv_2(self):
        OBJ = self.b(self.B)
        self.assertEqual(OBJ.name, self.B.name)
        self.assertEqual(OBJ.data, self.B.data)
        self.assertEqual(OBJ.unit, self.B.unit)
        self.assertIsInstance(OBJ, self.b)

    def test_1(self):
        OBJ = self.a(self.B)
        self.assertEqual(OBJ.name, self.B.name)
        self.assertEqual(OBJ.data, self.B.data)
        self.assertEqual(OBJ.unit, self.B.unit)
        self.assertIsInstance(OBJ, self.a)

    def test_2(self):
        OBJ = self.b(self.A)
        self.assertEqual(OBJ.name, self.A.name)
        self.assertEqual(OBJ.data, self.A.data)
        self.assertEqual(OBJ.unit, self.A.unit)
        self.assertIsInstance(OBJ, self.b)

    def test_3(self):
        OBJ = self.a(self.B, unit="Pa")
        self.assertEqual(OBJ.name, self.B.name)
        self.assertEqual(OBJ.data, self.B.data)
        self.assertEqual(OBJ.unit, "Pa")
        self.assertIsInstance(OBJ, self.a)

    def test_4(self):
        OBJ = self.b(self.A, unit="s")
        self.assertEqual(OBJ.name, self.A.name)
        self.assertEqual(OBJ.data, self.A.data)
        self.assertEqual(OBJ.unit, "s")
        self.assertIsInstance(OBJ, self.b)


class TestDimConv1(TestDimConv):
    a = ddim.DimSweep
    b = ddim.DimRep


class Object:
    pass


class TestDate(TestCase):
    def test_date_1(self):
        res = np.array(["2012-05-30 12:12:31", "2012-05-31 12:12:31"],
                       np.dtype("datetime64[us]"))
        d = DimSweep("a", res)
        self.assertEqual(d.data.dtype, np.dtype("datetime64[us]"))

    def test_obj(self):
        d = DimSweep("a", np.array([Object(), Object()]))
        self.assertEqual(d.data.dtype, np.object)

    def test_empty(self):
        d = DimSweep("a", [])
        self.assertEqual(d.data.dtype, np.float64)


class Testoutputformat(TestCase):
    def test_int(self):
        d = DimSweep("a", [1, 2, 3])
        self.assertEqual(d.outputformat, "%d")

    def test_num(self):
        d = DimSweep("a", [1., 2., 3])
        self.assertEqual(d.outputformat, "%.16e")

    def test_str(self):
        d = DimSweep("a", ["a"])
        self.assertEqual(d.outputformat, "%s")
