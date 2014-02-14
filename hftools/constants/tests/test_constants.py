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

import hftools.constants.si_units as si_units

from hftools.testing import TestCase, make_load_tests
from hftools.dataset import hfarray

load_tests = make_load_tests(si_units)
basepath = os.path.split(__file__)[0]

from hftools.constants import unit_to_multiplier


class Test_unit_to_multiplier(TestCase):
    unit = "Hz"

    def _helper(self, prefix, multiplier):
        res = si_units.unit_to_multiplier("%s%s" % (prefix, self.unit))
        self.assertEqual(res, (multiplier, self.unit))

    def test_a(self):
        self._helper("a", 1e-18)

    def test_f(self):
        self._helper("f", 1e-15)

    def test_p(self):
        self._helper("p", 1e-12)

    def test_n(self):
        self._helper("n", 1e-9)

    def test_u(self):
        self._helper("u", 1e-6)

    def test_m(self):
        self._helper("m", 1e-3)

    def test_U(self):
        self._helper("", 1)

    def test_k(self):
        self._helper("k", 1e3)

    def test_M(self):
        self._helper("M", 1e6)

    def test_G(self):
        self._helper("G", 1e9)

    def test_T(self):
        self._helper("T", 1e12)

    def test_1(self):
        self.assertEqual(unit_to_multiplier("s"), (1, "s"))

    def test_2(self):
        self.assertEqual(unit_to_multiplier("mXYZ"), (1, "mXYZ"))

    def test_3(self):
        self.assertEqual(unit_to_multiplier("%"), (.01, None))

    def test_4(self):
        self.assertEqual(unit_to_multiplier("#"), (1, None))


class Test_unit_to_multiplier_F(Test_unit_to_multiplier):
    unit = "F"


class Test_convert_with_unit(TestCase):

    def test_1(self):
        self.assertEqual(si_units.convert_with_unit("", .23),
                         hfarray(0.23, unit=None))


class Test_format(TestCase):
    def test_inf1(self):
        self.assertEqual(si_units.format_number(np.inf), "inf ")

    def test_inf2(self):
        self.assertEqual(si_units.format_number(-np.inf), "-inf ")

    def test1(self):
        res = si_units.format_number(12.345, unit="s", digs=0)
        self.assertEqual(res, "12 s")

    def test2(self):
        res = si_units.format_number(12.345, unit="m", digs=1)
        self.assertEqual(res, "12.3 m")

    def test3(self):
        res = si_units.format_number(12.345, unit="V", digs=2)
        self.assertEqual(res, "12.35 V")

    def test4(self):
        res = si_units.format_number(12.345, unit="A", digs=3)
        self.assertEqual(res, "12.345 A")


class Test_string_number_with_unit_to_value(TestCase):
    def test_1(self):
        sol = si_units.string_number_with_unit_to_value("1.2e2 V")
        self.assertAllclose(sol, hfarray(1.2e2, unit="V"))
        self.assertEqual(sol.unit, "V")

    def test_2(self):
        sol = si_units.string_number_with_unit_to_value("1.2e2")
        self.assertAllclose(sol, hfarray(1.2e2, unit=None))
        self.assertEqual(sol.unit, None)

    def test_3(self):
        sol = si_units.string_number_with_unit_to_value("13")
        self.assertAllclose(sol, hfarray(13, unit=None))
        self.assertEqual(sol.unit, None)

    def test_4(self):
        self.assertRaises(ValueError,
                          si_units.string_number_with_unit_to_value, "aa")

    def test_5(self):
        self.assertEqual(si_units.string_number_with_unit_to_value(5), 5)
        self.assertEqual(si_units.string_number_with_unit_to_value(5.), 5.)


class TestSIFormat(TestCase):
    def test1(self):
        fmt = si_units.SIFormat(unit="Hz")
        self.assertEqual(fmt % 12.345, "12.345 Hz")
        self.assertEqual(fmt % 12.345e6, "12.345 MHz")

    def test2(self):
        fmt = si_units.SIFormat(unit="Hz", digs=2)
        self.assertEqual(fmt % 12.345, "12.35 Hz")
        self.assertEqual(fmt % 12.346e6, "12.35 MHz")

if __name__ == '__main__':
    import unittest2 as unittest
    unittest.main()
