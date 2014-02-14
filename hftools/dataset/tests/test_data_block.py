#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os
import pdb
import warnings

import numpy as np

import hftools.dataset.dim as ddim
import hftools.dataset.dataset as dset
import hftools.dataset as ds

from hftools.dataset import hfarray, DimSweep, DimRep, DimMatrix_i,\
    DimMatrix_j, DataBlockError, DataDict, DataBlock
from hftools.dataset.comments import Comments
from hftools.testing import random_value_array, random_complex_value_array,\
    random_value_array_from_info, random_value_array, make_load_tests,\
    random_complex_value_array, random_value_array_from_info, TestCase
from hftools.utils import reset_hftools_warnings, HFToolsWarning

basepath = os.path.split(__file__)[0]
load_tests = make_load_tests(dset)

VA = hfarray
import hftools.dataset


class Test_DataBlock(TestCase):
    def setUp(self):
        self.a = DataBlock()

    def test_report(self):
        self.a.report()

    def test_str(self):
        self.assertEqual(self.a.report(), str(self.a))

    def test_guess_1(self):
        self.assertIsNone(self.a.guess_units(False))


class Test_DataBlock_replace_ivardata(TestCase):
    def test1(self):
        a = DataBlock()
        a.i = DimSweep("i", [1, 2, 3])

        def funk(a):
            info = (DimSweep("i", [1]), DimSweep("j", [1]))
            a.i = hfarray([[1]], dims=info)
        self.assertRaises(AttributeError, funk, a)


class Test_DataBlock2(Test_DataBlock):
    def setUp(self):
        Test_DataBlock.setUp(self)
        self.a.P = hfarray([0.5, -.3, .5])
        Sinfo = (DimMatrix_i("i", 2), DimMatrix_j("j", 2))
        self.a.S = hfarray([[11, 12], [21, 22]], dims=Sinfo)
        Winfo = (DimSweep("g", 1), DimMatrix_i("i", 2), DimMatrix_j("j", 2))
        self.a.W = hfarray([[[11, 12], [21, 22]]], dims=Winfo)
        self.a.T = hfarray(["Text"], dims=(DimSweep("text_i", ["index0"]),))

    def test_1(self):
        self.a.xname = "freq"
        f = self.a.xvalue
        self.assertAllclose(f, np.arange(3))

    def test_getattr_1(self):
        P = self.a.P
        self.assertAllclose(P, np.array([0.5, -.3, .5]))

    def test_delattr_1(self):
        del self.a.X
        del self.a.P
        self.assertRaises(AttributeError, lambda: self.a.P)

    def test_in_1(self):
        self.assertTrue("P" in self.a)
        self.assertTrue("freq" in self.a)
        self.assertFalse("X" in self.a)

    def test_getitem(self):
        self.assertAllclose(self.a["P"], [.5, -.3, .5])
        self.assertEqual(self.a["S11"], 11)
        self.assertRaises(KeyError, self.a.__getitem__, "P11")
        self.assertRaises(KeyError, self.a.__getitem__, "X11")

    def test_setitem_1(self):
        a = DataBlock()
        a.l = DimSweep("l", [.1, .2, .3])
        self.assertAllclose(a.l, [.1, .2, .3])
        self.assertAllclose(a.xvalue, [.1, .2, .3])
        self.assertEqual(a.xname, "l")
        a.w = DimSweep("w", [1.1, 1.2, 1.3])
        self.assertAllclose(a.w, [1.1, 1.2, 1.3])
        self.assertAllclose(a.xvalue, [.1, .2, .3])
        self.assertEqual(a.xname, "l")

    def test_setitem_2(self):
        self.a.W11 = hfarray([1])
        self.assertAllclose(self.a.W, [[[1, 12], [21, 22]]])

    def test_setitem_3(self):
        self.a.S11 = hfarray(1)
        self.assertAllclose(self.a.S, [[1, 12], [21, 22]])

    def test_setoutputformat_1(self):
        self.a.set_outputformat("P", "%.3f")
        self.assertEqual(self.a.P.outputformat, "%.3f")
        self.assertEqual(self.a.S.outputformat, "%d")
        self.assertEqual(self.a.W.outputformat, "%d")
        self.assertEqual(self.a.T.outputformat, "%s")

    def test_setoutputformat_2(self):
        self.a.set_outputformat("i", "%.3f")
        self.assertEqual(self.a.i.outputformat, "%.3f")
        self.assertEqual(self.a.P.outputformat, "%.16e")
        self.assertEqual(self.a.S.outputformat, "%d")
        self.assertEqual(self.a.W.outputformat, "%d")
        self.assertEqual(self.a.T.outputformat, "%s")

    def test_setoutputformat_error_1(self):
        self.assertRaises(DataBlockError,
                          self.a.set_outputformat, None, "%.3f")


class Test_DataBlock_rename(TestCase):
    def test_1(self):
        d = DataBlock()
        info = (DimSweep("Freq[Hz]", 3),)
        d.P = hfarray([1, 2, 3], dims=info)
        d.L = hfarray([10, 20, 30], dims=info)
        self.assertEqual(d.allvarnames, ["Freq[Hz]", "P", "L"])
        d.rename("P", "H")
        self.assertEqual(d.allvarnames, ["Freq[Hz]", "H", "L"])
        d.rename("Freq[Hz]", "f")
        self.assertEqual(d.allvarnames, ["f", "H", "L"])
        d.rename("foo", "bar")
        self.assertEqual(d.allvarnames, ["f", "H", "L"])

    def test_2(self):
        d = DataBlock()
        fi = DimSweep("Freq[Hz]", 3)
        gi = DimSweep("P[W]", 2)
        info = (fi,)
        info2 = (fi, gi)
        info3 = (gi,)
        d.P = hfarray([1, 2, 3], dims=info)
        d.Q = hfarray([3, 2], dims=info3)
        d.L = hfarray([[10, 20, 30], [11, 21, 31]], dims=info2)
        self.assertEqual(d.allvarnames, ["Freq[Hz]", "P[W]", "P", "Q", "L"])
        d.rename("P", "H")
        self.assertEqual(d.allvarnames, ["Freq[Hz]", "P[W]", "H", "Q", "L"])
        d.rename("Freq[Hz]", "f")
        self.assertEqual(d.allvarnames, ["f", "P[W]", "H", "Q", "L"])
        d.rename("foo", "bar")
        self.assertEqual(d.allvarnames, ["f", "P[W]", "H", "Q", "L"])


class Test_allvarnames(TestCase):
    def test_1(self):
        d = DataBlock()
        fi = DimSweep("Freq[Hz]", 3)
        gi = DimSweep("P[W]", 3)
        info = (fi,)
        info2 = (fi, gi)
        info3 = (gi,)
        d["P[W]"] = hfarray([10, 20, 30], dims=info)
        d["f"] = hfarray([[1, 2, 3]] * 3, dims=info2)
        self.assertEqual(d.allvarnames, ["Freq[Hz]", "P[W]", "f"])
        self.assertAllclose(d["P[W]"], [0, 1, 2])

    def test_2(self):
        d = DataBlock()
        fi = DimSweep("Freq[Hz]", 3)
        gi = DimSweep("P[W]", 3)
        info = (fi,)
        info2 = (fi, gi)
        info3 = (gi,)
        d["f"] = hfarray([[1, 2, 3]] * 3, dims=info2)
        d["P[W]"] = hfarray([10, 20, 30], dims=info)
        self.assertEqual(d.allvarnames, ["Freq[Hz]", "P[W]", "f"])
        self.assertAllclose(d["P[W]"], [10, 20, 30])


class Test_outputformat(TestCase):
    def test_1(self):
        d = DataBlock()
        fi = DimSweep("Freq[Hz]", 3)
        gi = DimSweep("P[W]", 3)
        info = (fi,)
        info2 = (fi, gi)
        d["f"] = hfarray([[1, 2, 3]] * 3, dims=info2, outputformat="%.5f")
        d["P"] = hfarray([10, 20, 30], dims=info, outputformat="%.7f")
        self.assertEqual(d["f"].outputformat, "%.5f")
        self.assertEqual(d["P"].outputformat, "%.7f")
        d.outputformat = "%.9f"
        for v in d.ivardata.values():
            self.assertEqual(v.outputformat, "%.9f")
        for v in d.vardata.values():
            self.assertEqual(v.outputformat, "%.9f")


class Test_copy_view(TestCase):
    def test_copy_1(self):
        d = DataBlock()
        fi = DimSweep("Freq[Hz]", 3)
        info = (fi,)
        d["a"] = hfarray([1, 2, 3], dims=info,
                            outputformat="%.5f", unit="V")
        d["b"] = hfarray([10, 20, 30], dims=info,
                            outputformat="%.7f", unit="A")
        c = d.copy()
        self.assertEqual(d.ivardata.order, c.ivardata.order)
        self.assertEqual(d.vardata.order, c.vardata.order)
        self.assertEqual(c.allvarnames, d.allvarnames)
        self.assertEqual(c.vardata.keys(), d.vardata.keys())
        self.assertEqual(c.ivardata.keys(), d.ivardata.keys())
        self.assertAllclose(c["Freq[Hz]"], d["Freq[Hz]"])
        self.assertAllclose(d.a, [1, 2, 3])
        self.assertAllclose(c.a, d.a)
        self.assertAllclose(c.b, d.b)
        c.a[0] = 7
        c.b[0] = 7
        self.assertAllclose(c.a, [7, 2, 3])
        self.assertAllclose(c.b, [7, 20, 30])
        self.assertAllclose(d.a, [1, 2, 3])
        self.assertAllclose(d.b, [10, 20, 30])

    def test_copy_2(self):
        d = DataBlock()
        fi = DimSweep("Freq[Hz]", 3)
        info = (fi,)
        d.comments = Comments(["Hej=10"])
        d["a"] = hfarray([1, 2, 3], dims=info,
                            outputformat="%.5f", unit="V")
        d["b"] = hfarray([10, 20, 30], dims=info,
                            outputformat="%.7f", unit="A")
        c = d.copy()
        self.assertEqual(c.allvarnames, d.allvarnames)
        self.assertEqual(c.vardata.keys(), d.vardata.keys())
        self.assertEqual(c.ivardata.keys(), d.ivardata.keys())
        self.assertAllclose(c["Freq[Hz]"], d["Freq[Hz]"])
        self.assertAllclose(d.a, [1, 2, 3])
        self.assertAllclose(c.a, d.a)
        self.assertAllclose(c.b, d.b)
        self.assertFalse(id(d.comments) == id(c.comments))
        c.a[0] = 7
        c.b[0] = 7
        self.assertAllclose(c.a, [7, 2, 3])
        self.assertAllclose(c.b, [7, 20, 30])
        self.assertAllclose(d.a, [1, 2, 3])
        self.assertAllclose(d.b, [10, 20, 30])

    def test_view_1(self):
        d = DataBlock()
        d.comments = Comments(["Hej=10"])
        fi = DimSweep("Freq[Hz]", 3)
        info = (fi,)
        d["a"] = hfarray([1, 2, 3], dims=info,
                            outputformat="%.5f", unit="V")
        d["b"] = hfarray([10, 20, 30], dims=info,
                            outputformat="%.7f", unit="A")
        c = d.view()
        self.assertEqual(c.allvarnames, d.allvarnames)
        self.assertEqual(c.vardata.keys(), d.vardata.keys())
        self.assertEqual(c.ivardata.keys(), d.ivardata.keys())
        self.assertAllclose(c["Freq[Hz]"], d["Freq[Hz]"])
        self.assertAllclose(c.a, d.a)
        self.assertAllclose(c.b, d.b)
        self.assertTrue(id(d.comments) == id(c.comments))
        c.a[0] = 7
        c.b[0] = 7
        self.assertAllclose(c.a, [7, 2, 3])
        self.assertAllclose(c.b, [7, 20, 30])
        self.assertAllclose(d.a, [7, 2, 3])
        self.assertAllclose(d.b, [7, 20, 30])


class Test_filter(TestCase):
    def test_1(self):
        d = DataBlock()
        fi = DimSweep("Freq[Hz]", 3)
        gi = DimSweep("g", 4)
        info = (fi,)
        d["a"] = hfarray([1, 2, 3], dims=info,
                            outputformat="%.5f", unit="V")
        d["b"] = hfarray([10, 20, 30], dims=info,
                            outputformat="%.7f", unit="A")
        d["c"] = hfarray([10, 20, 30, 40], dims=(gi,),
                            outputformat="%.7f", unit="A")

        w = d.filter(d["Freq[Hz]"] <= 1)
        self.assertEqual(id(d.comments), id(w.comments))
        self.assertEqual(w["Freq[Hz]"].info, (DimSweep("Freq[Hz]", 2),))
        self.assertEqual(w.a.info, (DimSweep("Freq[Hz]", 2),))
        self.assertEqual(w.b.info, (DimSweep("Freq[Hz]", 2),))
        self.assertEqual(w.c.info, (DimSweep("g", 4),))

        self.assertAllclose(w["Freq[Hz]"], [0, 1])
        self.assertAllclose(w.a, [1, 2])
        self.assertAllclose(w.b, [10, 20])
        self.assertAllclose(w.c, [10, 20, 30, 40])
        self.assertEqual(w.a.unit, "V")
        self.assertEqual(w.b.unit, "A")
        self.assertEqual(w.c.unit, "A")

    def test_2(self):
        d = DataBlock()
        fi = DimSweep("Freq[Hz]", 3)
        gi = DimSweep("g", 4)
        info = (fi, gi)
        d["a"] = hfarray([[1, 2, 3]] * 4, dims=info,
                            outputformat="%.5f", unit="V")
        self.assertRaises(ValueError, d.filter, d.a < 2)

    def test_intersection_1(self):
        d = DataBlock()
        fi = DimSweep("Freq[Hz]", [10, 20, 30, 40, 50])
        gi = DimSweep("g", 4)
        info = (fi, gi)
        d["a"] = hfarray([[1, 2, 3, 4]] * 5, dims=info,
                            outputformat="%.5f", unit="V")
        x = DimSweep("Freq", [20, 40])
        reset_hftools_warnings()
        self.assertHFToolsWarning(d.filter, hfarray(x))
        with warnings.catch_warnings() as log:
            warnings.resetwarnings()
            warnings.simplefilter("ignore", HFToolsWarning)
            d.filter(hfarray(x))
        reset_hftools_warnings()

    def test_intersection_2(self):
        d = DataBlock()
        fi = DimSweep("Freq[Hz]", [10, 20, 30, 40, 50])
        gi = DimSweep("g", 4)
        info = (fi, gi)
        d["a"] = hfarray([[1, 2, 3, 4]] * 5, dims=info,
                            outputformat="%.5f", unit="V")
        x = DimSweep("Freq[Hz]", [20, 40])
        dres = d.filter(hfarray(x))
        self.assertAllclose(dres["Freq[Hz]"], [20, 40])


class Test_sort(TestCase):
    def setUp(self):
        d = DataBlock()
        fi = DimSweep("Freq[Hz]", 3)
        gi = DimSweep("g", 4)
        info = (fi,)
        d["a"] = hfarray([3, 2, 1], dims=info,
                            outputformat="%.5f", unit="V")
        d["b"] = hfarray([30, 10, 20], dims=info,
                            outputformat="%.7f", unit="A")
        d["c"] = hfarray([30, 20, 10, 40], dims=(gi,),
                            outputformat="%.7f", unit="A")
        d["ac"] = d.a * d.c
        self.d = d

    def test_error_1(self):
        d = self.d
        self.assertRaises(ValueError, d.sort, hfarray(d.ac))

    def test_1(self):
        d = self.d
        result = d.sort(d.a)
        self.assertAllclose(result.a, [1, 2, 3])
        self.assertAllclose(result.b, [20, 10, 30])
        self.assertAllclose(result.c, [30, 20, 10, 40])
        facit = hfarray([[30, 20, 10, 40],
                            [60, 40, 20, 80],
                            [90, 60, 30, 120]], dims=d.ac.info)
        self.assertAllclose(result.ac, facit)

    def test_2(self):
        d = self.d
        reset_hftools_warnings()
        self.assertHFToolsWarning(d.sort, DimSweep("NONE", [1, 2, 3]))
        with warnings.catch_warnings() as log:
            warnings.resetwarnings()
            warnings.simplefilter("ignore", HFToolsWarning)
            d.sort(DimSweep("NONE", [1, 2, 3]))
        reset_hftools_warnings()


class Test_remove_rep(TestCase):
    def setUp(self):
        d = DataBlock()
        fi = DimSweep("Freq[Hz]", 3)
        gi = DimRep("g", 2)
        hi = DimRep("h", 3)
        ji = DimRep("j", 4)
        info = (fi,)
        d["a"] = hfarray([3, 2, 1], dims=info,
                            outputformat="%.5f", unit="V")
        d["b"] = hfarray([30, 10], dims=(gi,),
                            outputformat="%.7f", unit="A")
        d["c"] = hfarray([[30, 20, 10, 40], [20, 30, 40, 10]],
                            dims=(gi, hi), outputformat="%.7f", unit="A")
        self.d = d

    def test_1(self):
        d = self.d
        result = d.remove_rep()
        self.assertTrue("AllReps" in result.ivardata)


class Test_hyper(TestCase):
    def setUp(self):
        d = DataBlock()
        gi = DimSweep("Index", 6)
        ri = DimRep("Rep", 2)
        info = (gi,)
        d["a"] = hfarray([1, 1, 2, 2, 3, 3], dims=info,
                            outputformat="%.5f", unit="V")
        d["b"] = hfarray([10, 20, 10, 20, 10, 20],
                            dims=(gi,), outputformat="%.7f", unit="A")
        d["c"] = hfarray([10.1, 20.1, 10.2, 20.2, 10.3, 20.3],
                            dims=(gi,), outputformat="%.7f", unit="A")
        d["d"] = hfarray([17, 19], dims=(ri,),
                            outputformat="%.7f", unit="A")
        self.d = d

    def test_1(self):
        d = self.d
        result = d.hyper(["a", "b"], "Index")
        self.assertAllclose(result.a, [1, 2, 3])
        self.assertAllclose(result.b, [10, 20])
        self.assertAllclose(result.c, [[10.1, 20.1],
                                       [10.2, 20.2],
                                       [10.3, 20.3]])

    def test_2(self):
        d = self.d
        result = d.hyper(["a", "b"], d.ivardata["Index"])
        self.assertAllclose(result.a, [1, 2, 3])
        self.assertAllclose(result.b, [10, 20])
        self.assertAllclose(result.c, [[10.1, 20.1],
                                       [10.2, 20.2],
                                       [10.3, 20.3]])

    def test_3(self):
        d = self.d
        result = d.hyper(["a", "b"], "Index", all=True, indexed=True)
        self.assertAllclose(result.a, [[1, 1], [2, 2], [3, 3]])
        self.assertAllclose(result.b, [[10, 20], [10, 20], [10, 20]])
        self.assertAllclose(result.a_index, [1, 2, 3])
        self.assertAllclose(result.b_index, [10, 20])
        self.assertAllclose(result.c, [[10.1, 20.1],
                                       [10.2, 20.2], [10.3, 20.3]])
        self.assertAllclose(result.d, [17, 19])

    def test_4(self):
        self.d.x = hfarray(1)
        result = self.d.hyper(["a", "b"], "Index", all=False)
        self.assertTrue("x" not in result)

    def test_5(self):
        self.d.x = hfarray(1)
        result = self.d.hyper(["a", "b"], "Index", all=True)
        self.assertTrue("x" in result)

    def test_6(self):
        self.d.y = DimSweep("y", [1, 2])
        result = self.d.hyper(["a", "b"], "Index", all=True)
        self.assertTrue("y" in result)

class Test_guess_units(TestCase):
    def test_1(self):
        d = DataBlock()
        fi = DimSweep("Freq", 3)
        gi = DimSweep("g", 4)
        d["Vds"] = hfarray([1, 2, 3], dims=(fi,))
        d["Id"] = hfarray([1, 2, 3, 4], dims=(gi,))
        d.guess_units()
        d.values_from_property()

        self.assertEqual(d.Vds.unit, "V")
        self.assertAllclose(d.Vds, [1, 2, 3])
        self.assertEqual(d.Id.unit, "A")
        self.assertAllclose(d.Id, [1, 2, 3, 4])
        self.assertEqual(d.Freq.unit, "Hz")

    def test_2(self):
        d = DataBlock()
        fi = DimSweep("Freq", 3, unit="m")
        gi = DimSweep("g", 4)
        d["Vds"] = hfarray([1, 2, 3], dims=(fi,), unit="A")
        d["Id"] = hfarray([1, 2, 3, 4], dims=(gi,), unit="V")
        d.guess_units()

        self.assertEqual(d.Vds.unit, "A")
        self.assertAllclose(d.Vds, [1, 2, 3])
        self.assertEqual(d.Id.unit, "V")
        self.assertAllclose(d.Id, [1, 2, 3, 4])
        self.assertEqual(d.Freq.unit, "m")

    def test_3(self):
        d = DataBlock()
        fi = DimSweep("Freq", 3)
        gi = DimSweep("g", 4)
        d["Vds"] = hfarray([1, 2, 3], dims=(fi,))
        d["Id"] = hfarray([1, 2, 3, 4], dims=(gi,))
        d.guess_units("Freq")

        self.assertEqual(d.Vds.unit, None)
        self.assertAllclose(d.Vds, [1, 2, 3])
        self.assertEqual(d.Id.unit, None)
        self.assertAllclose(d.Id, [1, 2, 3, 4])
        self.assertEqual(d.Freq.unit, "Hz")

    def test_4(self):
        d = DataBlock()
        fi = DimSweep("Freq", 3)
        gi = DimSweep("g", 4)
        d["Vds"] = hfarray([1, 2, 3], dims=(fi,))
        d["Id"] = hfarray([1, 2, 3, 4], dims=(gi,))
        d.guess_units(["Freq", "Vds", "Ig"])

        self.assertEqual(d.Vds.unit, "V")
        self.assertAllclose(d.Vds, [1, 2, 3])
        self.assertEqual(d.Id.unit, None)
        self.assertAllclose(d.Id, [1, 2, 3, 4])
        self.assertEqual(d.Freq.unit, "Hz")

    def test_5(self):
        d = DataBlock()
        d.comments = Comments(["Vgs=13", "Ig=14", "Datetime=2011-10-11 20:11:02",
                               "Vds=-13", "Calibration=SOLT"])
        fi = DimSweep("Freq", 3)
        gi = DimSweep("g", 4)
        d["Vds"] = hfarray([1, 2, 3], dims=(fi,))
        d["Id"] = hfarray([1, 2, 3, 4], dims=(gi,))
        d.comments.property["Ig"].unit = "mA"
        d.guess_units()
        d.values_from_property()

        self.assertEqual(d.Vds.unit, "V")
        self.assertAllclose(d.Vds, [1, 2, 3])
        self.assertEqual(d.Id.unit, "A")
        self.assertAllclose(d.Id, [1, 2, 3, 4])
        self.assertEqual(d.Freq.unit, "Hz")
        self.assertEqual(d.comments.property["Vgs"].unit, "V")
        self.assertEqual(d.comments.property["Ig"].unit, "mA")

        self.assertEqual(d.Ig, 14)
        self.assertEqual(d.Vgs, 13)
        self.assertEqual(d.Ig.unit, "mA")
        self.assertEqual(d.Vgs.unit, "V")
        self.assertTrue("Datetime" in d.allvarnames)


class Test_replace_dim(TestCase):
    def setUp(self):
        self.d = DataBlock()
        self.fi = DimSweep("Freq", 3)
        self.gi = DimSweep("g", 4)
        self.hi = DimSweep("h", 4)
        self.d["Vds"] = hfarray([1, 2, 3], dims=(self.fi,))
        self.d["Id"] = hfarray([1, 2, 3, 4], dims=(self.gi,))

    def test_error_1(self):
        self.assertRaises(KeyError, self.d.replace_dim, self.hi, self.fi)

    def test_error_2(self):
        self.assertRaises(KeyError,
                          self.d.replace_dim, "nonexisting-dim", self.fi)

    def test_1(self):
        self.d.replace_dim("g", self.hi)
        self.assertTrue(self.hi not in self.d["Vds"].info)
        self.assertTrue(self.hi in self.d["Id"].info)

    def test_2(self):
        self.d.replace_dim("g", DimRep)
        self.assertIsInstance(self.d.Id.info[0], DimRep)
        self.assertEqual(self.d.Id.info[0].name, "g")


class Test_blockname(TestCase):
    def test_1(self):
        d = DataBlock()
        d.blockname = None
        d.FILENAME = hfarray(["foo.txt", "bar.txt"],
                                dims=(DimRep("FILEINDEX", [0, 1]),))
        self.assertEqual(d.blockname, "foo.txt")

    def test_2(self):
        d = DataBlock()
        d.FILENAME = hfarray(["foo.txt", "bar.txt"],
                                dims=(DimRep("FILEINDEX", [0, 1]),))
        d.blockname = "baz.txt"
        self.assertEqual(d.blockname, "baz.txt")

class Test_subset_datablock_by_dims(TestCase):
    def test_error(self):
        fi = DimSweep("freq", [1,])
        db = DataBlock()
        self.assertRaises(ValueError, dset.subset_datablock_by_dims, db, (fi, fi))

    def test_1(self):
        db = DataBlock()
        Sinfo = (DimRep("freq", [1, 2]) ,DimRep("r", [1]), DimMatrix_i("i", 2), DimMatrix_j("j", 2))
        db.S = hfarray(np.array([[11, 12], [21, 22]])[np.newaxis, np.newaxis, :, :] * np.array([[10], [20]])[..., np.newaxis, np.newaxis], dims=Sinfo)
        db.V = hfarray([1.23], Sinfo[1:2])
        db.Y = hfarray([1.23], (DimRep("k", [1]),))
        out = dset.subset_datablock_by_dims(dset.convert_matrices_to_elements(db), Sinfo[:-2])
        self.assertTrue("V" in out)
        self.assertTrue("S11" in out)
        self.assertTrue("S12" in out)
        self.assertTrue("S21" in out)
        self.assertTrue("S22" in out)
        self.assertFalse("Y" in out)

class Test_convert_matrices_to_elements(TestCase):
    def setUp(self):
        self.db = db = DataBlock()
        Sinfo = (DimRep("freq", [1, 2]) ,DimRep("r", [1]), DimMatrix_i("i", 2), DimMatrix_j("j", 2))
        db.S = hfarray(np.array([[11, 12], [21, 22]])[np.newaxis, np.newaxis, :, :] * np.array([[10], [20]])[..., np.newaxis, np.newaxis], dims=Sinfo)
        db.V = hfarray([1.23], Sinfo[1:2])
        db.Y = hfarray([1.23], (DimRep("k", [1]),))

    def test_1(self):
        out = dset.convert_matrices_to_elements(self.db)
        self.assertTrue("V" in out)
        self.assertTrue("S11" in out)
        self.assertTrue("S12" in out)
        self.assertTrue("S21" in out)
        self.assertTrue("S22" in out)
        self.assertTrue("Y" in out)

    def test_2(self):
        def formatelement(varname, i, j):
            return "%s%s%s"%(varname, i ,j)
        out = dset.convert_matrices_to_elements(self.db, formatelement)
        self.assertTrue("V" in out)
        self.assertTrue("S11" in out)
        self.assertTrue("S12" in out)
        self.assertTrue("S21" in out)
        self.assertTrue("S22" in out)
        self.assertTrue("Y" in out)

class Test_yield_dim_consistent_datablocks(TestCase):
    def setUp(self):
        self.db = db = DataBlock()
        Sinfo = (DimRep("freq", [1, 2]) ,DimRep("r", [1]), DimMatrix_i("i", 2), DimMatrix_j("j", 2))
        db.S = hfarray(np.array([[11, 12], [21, 22]])[np.newaxis, np.newaxis, :, :] * np.array([[10], [20]])[..., np.newaxis, np.newaxis], dims=Sinfo)
        db.V = hfarray([1.23], Sinfo[1:2])
        db.Y = hfarray([1.23], (DimRep("k", [1]),))

    def test1(self):
        res = list(dset.yield_dim_consistent_datablocks(dset.convert_matrices_to_elements(self.db)))
        self.assertTrue(len(res) == 2)


if __name__ == '__main__':
        d = DataBlock()
        d.comments = Comments(["Vgs=10", "Ig=10"])
        fi = DimSweep("Freq[GHz]", 3)
        gi = DimSweep("g", 4)
        d["Vds"] = hfarray([1, 2, 3], dims=(fi,))
        d["Id"] = hfarray([1, 2, 3, 4], dims=(gi,))
        d.guess_units()
