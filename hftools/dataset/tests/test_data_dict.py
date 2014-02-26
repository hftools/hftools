#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os
import pdb

import numpy as np
import unittest2 as unittest

import hftools.dataset.dim as ddim
import hftools.dataset.dataset as dset
import hftools.dataset as ds
from hftools.testing import random_value_array, random_complex_value_array,\
    random_value_array_from_dims
from hftools.dataset import hfarray, DimSweep, DimRep
from hftools.testing import TestCase, make_load_tests

from hftools.dataset import DataDict

basepath = os.path.split(__file__)[0]
load_tests = make_load_tests(dset)

VA = hfarray


class Test_DataDict(TestCase):
    def setUp(self):
        self.a = DataDict()
        self.b = DataDict(a=VA(1), c=VA(3), b=VA(2), d=VA(4))
        self.c = DataDict(e=VA(12), a=VA(4))

    def _outputformat(self, var, finaloutformat):
        v = getattr(self, var)
        v.outputformat = "%.0f"
        for x in v.values():
            self.assertEqual(x.outputformat, "%.0f")
        self.assertEqual(v.outputformat, finaloutformat)

    def test_outputformat_1(self):
        self._outputformat("a", "%.16e")

    def test_outputformat_2(self):
        self._outputformat("b", "%.0f")

    def test_outputformat_3(self):
        self._outputformat("c", "%.0f")

    def test_outputformat_4(self):
        self.a["q"] = 10
        self.assertEqual(self.a.outputformat, "%.16e")

    def test_order_1(self):
        self.assertEqual(self.a.order, [])
        self.assertEqual(self.b.order, [])
        self.assertEqual(self.c.order, [])

    def test_order_2(self):
        self.b.order = ["a", "c", "b"]
        self.c.order = ["e"]
        self.assertEqual(self.a.keys(), [])
        self.assertEqual(self.b.keys(), ["a", "c", "b", "d"])
        self.assertEqual(self.c.keys(), ["e", "a"])

    def test_order_3(self):
        a = DataDict()
        a["a"] = 10
        a["d"] = 10
        a["g"] = 10
        a["c"] = 10
        self.assertEqual(a.keys(), ["a", "d", "g", "c"])

    def test_attr_getter_1(self):
        self.assertRaises(AttributeError, lambda: self.a.a)

    def test_attr_getter_2(self):
        self.assertRaises(AttributeError, lambda: self.b.e)
        self.assertEqual(self.b.a, 1)
        self.assertEqual(self.b.b, 2)
        self.assertEqual(self.b.c, 3)
        self.assertEqual(self.b.d, 4)

    def test_attr_getter_3(self):
        self.assertRaises(AttributeError, lambda: self.c.b)
        self.assertEqual(self.c.a, 4)
        self.assertEqual(self.c.e, 12)

    def test_attr_del_1(self):
        self.b.order = ["a", "c", "b"]
        del self.b.a
        self.assertDictEqual(self.b, dict(c=VA(3), b=VA(2), d=VA(4)))
        self.assertEqual(self.b.keys(), ["c", "b", "d"])
        del self.b.c
        self.assertDictEqual(self.b, dict(b=VA(2), d=VA(4)))
        self.assertEqual(self.b.keys(), ["b", "d"])
        del self.b.d
        self.assertDictEqual(self.b, dict(b=VA(2), ))
        self.assertEqual(self.b.keys(), ["b"])
        del self.b.b
        self.assertDictEqual(self.b, dict())
        self.assertEqual(self.b.keys(), [])

    def test_delitem_1(self):
        del self.a["k"]
        self.assertDictEqual(self.a, dict())

    def test_delitem_2(self):
        c = self.c
        c.order = ["e"]
        del c["k"]
        self.assertEqual(c.keys(), ["e", "a"])
        self.assertDictEqual(c, dict(e=VA(12), a=VA(4)))
        del c["e"]
        self.assertEqual(c.keys(), ["a"])
        self.assertDictEqual(c, dict(a=VA(4)))
        del c["a"]
        self.assertEqual(c.keys(), [])
        self.assertDictEqual(c, dict())

    def test_set_attr_1(self):
        a = self.a
        a.keys
        a.order = []
        a.a = VA(10)
        self.assertEqual(a.keys(), ["a"])
        self.assertDictEqual(a, dict(a=VA(10)))
        a.v = VA(22)
        self.assertEqual(a.keys(), ["a", "v"])
        self.assertDictEqual(a, dict(a=VA(10), v=VA(22)))
        a.a = VA(12)
        self.assertEqual(a.keys(), ["a", "v"])
        self.assertDictEqual(a, dict(a=VA(12), v=VA(22)))

    def test_set_default_1(self):
        a = self.a
        p = a.setdefault("a", VA(10))
        self.assertEqual(p, VA(10))
        self.assertEqual(a.keys(), ["a"])
        self.assertDictEqual(a, dict(a=VA(10)))
        p = a.setdefault("v", VA(22))
        self.assertEqual(p, VA(22))
        self.assertEqual(a.keys(), ["a", "v"])
        self.assertDictEqual(a, dict(a=VA(10), v=VA(22)))
        p = a.setdefault("a", VA(22))
        self.assertEqual(p, VA(10))
        self.assertEqual(a.keys(), ["a", "v"])
        self.assertDictEqual(a, dict(a=VA(10), v=VA(22)))

    def test_rename_1(self):
        b = self.b
        b.order = ["a", "c", "b"]
        b.rename("a", "A")
        self.assertDictEqual(b, dict(A=VA(1), c=VA(3), b=VA(2), d=VA(4)))
        self.assertEqual(b.keys(), ["A", "c", "b", "d"])

    def test_view_1(self):
        a = DataDict(a=VA([11, 17]), b=VA([7, 5, 23]))
        v = a.view()
        self.assertNotEqual(id(a.a), id(v.a))
        self.assertNotEqual(id(a.b), id(v.b))
        self.assertAllclose(a.a, v.a)
        self.assertAllclose(a.b, v.b)
        v.a[0] = 9
        v.b[0] = 90
        self.assertAllclose(v.a, VA([9, 17]))
        self.assertAllclose(v.b, VA([90, 5, 23]))
        self.assertAllclose(a.a, v.a)
        self.assertAllclose(a.b, v.b)

    def test_copy_1(self):
        a = DataDict(a=VA([11, 17]), b=VA([7, 5, 23]))
        v = a.copy()
        self.assertEqual(a.order, v.order)
        self.assertNotEqual(id(a.a), id(v.a))
        self.assertNotEqual(id(a.b), id(v.b))
        self.assertAllclose(a.a, v.a)
        self.assertAllclose(a.b, v.b)
        v.a[0] = 9
        v.b[0] = 90
        self.assertAllclose(v.a, VA([9, 17]))
        self.assertAllclose(v.b, VA([90, 5, 23]))
        self.assertAllclose(a.a, VA([11, 17]))
        self.assertAllclose(a.b, VA([7, 5, 23]))

    def test_items_1(self):
        c = self.c
        c.order = ["e"]
        self.assertEqual(c.items(), [("e", VA(12)), ("a", VA(4))])

    def test_iteritems_1(self):
        c = self.c
        c.order = ["e"]
        self.assertEqual(list(c.iteritems()), [("e", VA(12)), ("a", VA(4))])

    def test_values_1(self):
        c = self.c
        c.order = ["e"]
        self.assertEqual(c.values(), [VA(12), VA(4)])
