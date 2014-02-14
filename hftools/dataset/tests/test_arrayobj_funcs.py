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

