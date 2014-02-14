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

import hftools.constants.waveguide as wg

from hftools.testing import TestCase, make_load_tests

#uncomment when adding doctests
#load_tests = make_load_tests(wg)
basepath = os.path.split(__file__)[0]


class Test_WR(TestCase):
    unit = "Hz"

    def test_1(self):
        facit = wg.WaveGuide("WR650", "WG6", "R14", 1.15e9,
                             1.72e9, 0.908e9, 1.816e9, 6.5 * 25.4, 3.25 * 25.4)
        self.assertEqual(wg.WR["WR650"], facit)
