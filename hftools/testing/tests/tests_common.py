# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os

import numpy as np

import hftools.utils as utils
import hftools.testing as testing
from hftools.dataset import hfarray
from hftools.testing import TestCase, make_load_tests, expectedFailure

load_tests = make_load_tests(utils)
basepath = os.path.split(__file__)[0]


class Test_random_array(TestCase):
    def test_float_1(self):
        res = testing.random_array(5, 7, 2)
        self.assertEqual(res.ndim, 5)
        self.assertTrue(np.all([x >= 2 and x <= 7 for x in res.shape]))
        self.assertFalse(np.iscomplexobj(res), msg=str(res))
        self.assertIsNotInstance(res, hfarray)


class Test_random_complex_array(TestCase):
    def test_float_1(self):
        res = testing.random_complex_array(5, 7, 2)
        self.assertEqual(res.ndim, 5)
        self.assertTrue(np.all([x >= 2 and x <= 7 for x in res.shape]))
        self.assertTrue(np.iscomplexobj(res))
        self.assertIsNotInstance(res, hfarray)


class Test_random_complex_value_array(TestCase):
    def test_float_1(self):
        res = testing.random_complex_value_array(5, 7, 2)
        self.assertEqual(res.ndim, 5)
        self.assertTrue(np.all([x >= 2 and x <= 7 for x in res.shape]))
        self.assertTrue(np.iscomplexobj(res))
        self.assertIsInstance(res, hfarray)


class Test_random_complex_matrix(TestCase):
    def test_float_1(self):
        res = testing.random_complex_matrix(5, 7, 2, 2)
        self.assertEqual(res.ndim, 5 + 2)
        self.assertTrue(np.all([x >= 2 and x <= 7 for x in res.shape[:-2]]))
        self.assertTrue(np.iscomplexobj(res))
        self.assertIsInstance(res, hfarray)


class Test_testcase(TestCase):
    @expectedFailure
    def test1(self):  # Testcase to get coverage of failing path
        self.assertIsInstance("a", int)
