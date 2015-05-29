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

import numpy as np

import hftools.dataset.helper as helper

from hftools.testing import TestCase, make_load_tests
from hftools.dataset import hfarray

#uncomment to enable doctests
#load_tests = make_load_tests(helper)
basepath = os.path.split(__file__)[0]

from hftools.constants import unit_to_multiplier


class Test_add_var_guess(TestCase):
    def setUp(self):
        self.old = helper._varname_unit_guess_db.copy()

    def tearDown(self):
        helper._varname_unit_guess_db = self.old

    def test_1(self):
        helper.add_var_guess("R", "Ohm")
        self.assertIn("R", helper._varname_unit_guess_db)
        self.assertEqual("Ohm", helper._varname_unit_guess_db["R"])


class Test_guess_unit_from_varname(TestCase):
    def setUp(self):
        self.old = helper._varname_unit_guess_db.copy()

    def tearDown(self):
        helper._varname_unit_guess_db = self.old

    def test_1(self):
        unit = helper.guess_unit_from_varname("Vds")
        self.assertEqual(unit, "V")
