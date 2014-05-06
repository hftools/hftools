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

from numpy import newaxis

import hftools.dataset.arrayobj as aobj
import hftools.dataset.dim as dim
import hftools.dataset as ds

from hftools.testing import TestCase, skip, make_load_tests
from hftools.testing import random_value_array, random_complex_value_array,\
    SKIP

basepath = os.path.split(__file__)[0]


class Test_get_new_anonymous(TestCase):
    def test_get_new_anon_1(self):
        dims = (dim.DimSweep("a", 3),)
        anon = aobj.get_new_anonymous_dim(dims, [1, 2, 3])
        self.assertIsInstance(anon, aobj.DimAnonymous)
        self.assertEqual(anon.name, "ANON1")

    def test_get_new_anon_2(self):
        dims = (dim.DimSweep("ANON1", 3), dim.DimSweep("ANON2", 3), )
        anon = aobj.get_new_anonymous_dim(dims, [1, 2, 3])
        self.assertIsInstance(anon, aobj.DimAnonymous)
        self.assertEqual(anon.name, "ANON1")

    def test_get_new_anon_3(self):
        dims = aobj.hfarray([1, 2, 3])
        anon = aobj.get_new_anonymous_dim(dims, [1, 2, 3])
        self.assertIsInstance(anon, aobj.DimAnonymous)
        self.assertEqual(anon.name, "ANON1")


class Test_axis_handler(TestCase):
    def setUp(self):
        (ai, bi, ci) = (dim.DimSweep("ai", 2),
                        dim.DimRep("bi", 3),
                        dim.DimSweep("ci", 4))
        self.ai = ai
        self.bi = bi
        self.ci = ci
        self.a = aobj.hfarray(ai)
        self.b = aobj.hfarray(bi) * 10
        self.c = aobj.hfarray(ci) * 100
        self.abc = self.a + self.c + self.b

    def test_1(self):
        self.assertIsNone(aobj.axis_handler(self.a, None))

    def test_2(self):
        res = aobj.axis_handler(self.a, "ai")
        self.assertEqual(res, self.ai)

    def test_3(self):
        res = aobj.axis_handler(self.abc, 2)
        self.assertEqual(res, self.bi)

    def test_4(self):
        res = aobj.axis_handler(self.abc, dim.DimRep)
        self.assertEqual(res, self.bi)

    def test_5(self):
        res = aobj.axis_handler(self.abc, self.bi)
        self.assertEqual(res, self.bi)

    def test_error_1(self):
        self.assertRaises(IndexError, aobj.axis_handler,
                          self.abc, dim.DimAnonymous)

    def test_error_2(self):
        self.assertRaises(IndexError, aobj.axis_handler,
                          self.abc, dim.DimSweep)

    def test_error_3(self):
        self.assertRaises(IndexError, aobj.axis_handler,
                          self.a, self.ci)


class Test_multiple_axis_handler(TestCase):
    def setUp(self):
        (ai, bi, ci) = (dim.DimSweep("ai", 2),
                        dim.DimRep("bi", 3),
                        dim.DimSweep("ci", 4))
        self.ai = ai
        self.bi = bi
        self.ci = ci
        self.a = aobj.hfarray(ai)
        self.b = aobj.hfarray(bi) * 10
        self.c = aobj.hfarray(ci) * 100
        self.abc = self.a + self.b + self.c

    def test_1(self):
        self.assertEqual(aobj.multiple_axis_handler(self.a, None),
                         (None, None))

    def test_2(self):
        self.assertEqual(aobj.multiple_axis_handler(self.a, self.ai),
                         ((self.ai,), (0,)))

    def test_3(self):
        self.assertEqual(aobj.multiple_axis_handler(self.a, (self.ai, )),
                         ((self.ai,), (0,)))

    def test_4(self):
        self.assertEqual(aobj.multiple_axis_handler(self.a, (0, )),
                         ((self.ai,), (0,)))

    def test_5(self):
        self.assertEqual(aobj.multiple_axis_handler(self.abc, (0, )),
                         ((self.ai,), (0,)))

    def test_6(self):
        self.assertEqual(aobj.multiple_axis_handler(self.abc, ("ai", 1)),
                         ((self.ai, self.ci), (0, 1)))

    def test_7(self):
        self.assertEqual(aobj.multiple_axis_handler(self.abc, dim.DimSweep),
                         ((self.ai, self.ci), (0, 1)))

    def test_8(self):
        self.assertEqual(aobj.multiple_axis_handler(self.abc,
                                                    (self.ai, self.ci)),
                         ((self.ai, self.ci), (0, 1)))

    def test_erro_1(self):
        self.assertRaises(IndexError, aobj.multiple_axis_handler,
                          self.a, self.ci)
