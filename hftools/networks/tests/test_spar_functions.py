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
import hftools.networks.spar_functions as spfun

from hftools.testing import TestCase


basepath = os.path.split(__file__)[0]


def make_array(a):
    dims = (aobj.DimSweep("f", len(a)),
            aobj.DimMatrix_i("i", 2),
            aobj.DimMatrix_j("j", 2))
    return aobj.hfarray(a, dims=dims)


class Test_cascade(TestCase):
    def setUp(self):
        self.a = make_array([[[0, 1], [1, 0j]]])
        self.b = make_array([[[0, 2], [2, 0j]]])
        self.c = make_array([[[0.1, 0j], [0, 0.1]]])

    def test_cascade_1(self):
        r = spfun.cascadeS(self.a, self.a)
        self.assertTrue(np.allclose(r, self.a))

    def test_cascade_2(self):
        r = spfun.cascadeS(self.a, self.b)
        self.assertTrue(np.allclose(r, self.b))

    def test_cascade_3(self):
        r = spfun.cascadeS(self.b, self.b)
        self.assertTrue(np.allclose(r, self.a * 4))

    def test_cascade_4(self):
        r = spfun.cascadeS(self.a, self.c)
        self.assertTrue(np.allclose(r, self.c))

    def test_cascade_5(self):
        r = spfun.cascadeS(self.b, self.c)
        self.assertTrue(np.allclose(r, make_array([[[0.4, 0j], [0, 0.1]]])))

    def test_cascade_6(self):
        r = spfun.cascadeS(self.c, self.b)
        self.assertTrue(np.allclose(r, make_array([[[0.1, 0j], [0, 0.4]]])))


class Test_deembedleft(TestCase):
    def setUp(self):
        self.a = make_array([[[0, 1], [1, 0j]]])
        self.b = make_array([[[0, 2], [2, 0j]]])
        self.c = make_array([[[0.1, 0j], [0, 0.1]]])

    def test_cascade_1(self):
        r = spfun.deembedleft(self.a, self.a)
        self.assertTrue(np.allclose(r, self.a))

    def test_cascade_2(self):
        r = spfun.deembedleft(self.b, self.b)
        self.assertTrue(np.allclose(r, self.a))

    def test_cascade_3(self):
        r = spfun.deembedleft(self.b, self.c)
        self.assertTrue(np.allclose(r, make_array([[[0.025, 0j], [0, 0.1]]])))


class Test_deembedright(TestCase):
    def setUp(self):
        self.a = make_array([[[0, 1], [1, 0j]]])
        self.b = make_array([[[0, 2], [2, 0j]]])
        self.c = make_array([[[0.1, 0j], [0, 0.1]]])

    def test_cascade_1(self):
        r = spfun.deembedright(self.a, self.a)
        self.assertTrue(np.allclose(r, self.a))

    def test_cascade_2(self):
        r = spfun.deembedright(self.b, self.b)
        self.assertTrue(np.allclose(r, self.a))

    def test_cascade_3(self):
        r = spfun.deembedright(self.c, self.b)
        self.assertTrue(np.allclose(r, make_array([[[0.1, 0j], [0, 0.025]]])))


class Test_deembed(TestCase):
    def setUp(self):
        self.a = make_array([[[0, 1], [1, 0j]]])
        self.b = make_array([[[0, 2], [2, 0j]]])
        self.c = make_array([[[0.1, 0j], [0, 0.1]]])

    def test_cascade_1(self):
        r = spfun.deembed(self.a, self.a, self.a)
        self.assertTrue(np.allclose(r, self.a))

    def test_cascade_2(self):
        r = spfun.deembed(self.b, self.b, self.a)
        self.assertTrue(np.allclose(r, self.a))

    def test_cascade_3(self):
        r = spfun.deembed(self.a, self.b, self.b)
        self.assertTrue(np.allclose(r, self.a))

    def test_cascade_4(self):
        r = spfun.deembed(self.b, self.c, self.b)
        self.assertAllclose(r,  make_array([[[0.025, 0j], [0, 0.025]]]))

if __name__ == '__main__':
    unittest.main()
