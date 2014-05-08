# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import datetime as datetimemodule
import os
import pdb
import shutil

import numpy as np

import hftools.utils as utils
from hftools.testing import TestCase, make_load_tests

load_tests = make_load_tests(utils)
basepath = os.path.split(__file__)[0]


class Test_to_numeric(TestCase):
    def test_float_1(self):
        self.assertEqual(utils.to_numeric("1.2"), 1.2)

    def test_int_1(self):
        self.assertEqual(utils.to_numeric("1"), 1)

    def test_cplx_1(self):
        self.assertEqual(utils.to_numeric("1+1j", False), "1+1j")

    def test_date_1(self):
        self.assertEqual(utils.to_numeric("2012-05-30 10:34:00"),
                         np.datetime64("2012-05-30 10:34:00"))

    def test_date_2(self):
        self.assertEqual(utils.to_numeric("2012-05-30 10:34"),
                         np.datetime64("2012-05-30 10:34:00"))

    def test_nonnumeric_1(self):
        self.assertEqual(utils.to_numeric("a1.2", False), "a1.2")

    def test_nonnumeric_2(self):
        self.assertEqual(utils.to_numeric("", False), "")

    def test_raises_1(self):
        self.assertRaises(ValueError, utils.to_numeric, "", True)

    def test_raises_2(self):
        self.assertRaises(ValueError, utils.to_numeric, "")


class Test_stable_uniq(TestCase):
    def test_list_1(self):
        self.assertEqual(utils.stable_uniq([1, 1, 2, 2]), [1, 2])

    def test_list_2(self):
        self.assertEqual(utils.stable_uniq([2, 1, 3, 2, 1, 3]), [2, 1, 3])

    def test_array_1(self):
        self.assertEqual(utils.stable_uniq(np.array([1, 1, 2, 2])), [1, 2])

    def test_array_2(self):
        res = utils.stable_uniq(np.array([2, 1, 3, 2, 1, 3]))
        self.assertEqual(res, [2, 1, 3])

    def test_list_empty_1(self):
        self.assertEqual(utils.stable_uniq([]), [])


class Test_uniq(TestCase):
    def test_list_1(self):
        self.assertEqual(utils.uniq([1, 1, 2, 2]), [1, 2])

    def test_list_2(self):
        self.assertEqual(utils.uniq([2, 1, 3, 2, 1, 3]), [1, 2, 3])

    def test_array_1(self):
        self.assertEqual(utils.uniq(np.array([1, 1, 2, 2])), [1, 2])

    def test_array_2(self):
        self.assertEqual(utils.uniq(np.array([2, 1, 3, 2, 1, 3])), [1, 2, 3])

    def test_list_empty_1(self):
        self.assertEqual(utils.uniq([]), [])


class Test_chop(TestCase):
    def test_1(self):
        self.assertAllclose(utils.chop([1e-16]), np.array([0]))


class Test_lex_order(TestCase):
    def test_1(self):
        self.assertEqual(utils.lex_order(["bb", "aa"]), ["aa", "bb"])

    def test_2(self):
        res = utils.lex_order(["bb5", "bb11", "aa"])
        self.assertEqual(res, ["aa", "bb5", "bb11"])


class Test_make_dirs(TestCase):
    def setUp(self):
        bpath = os.path.split(os.path.abspath(__file__))[0]
        self.base = base = os.path.join(bpath, "XTEMPX")
        utils.make_dirs(base, "aa")
        utils.make_dirs(base, "bb5")
        utils.make_dirs(base, "bb11")

    def tearDown(self):
        shutil.rmtree(self.base)

    def test_1(self):
        self.assertTrue(os.path.isdir(self.base))

    def test_2(self):
        self.assertTrue(os.path.isdir(os.path.join(self.base, "aa")))

    def test_3(self):
        self.assertTrue(os.path.isdir(os.path.join(self.base, "bb5")))

    def test_4(self):
        self.assertTrue(os.path.isdir(os.path.join(self.base, "bb11")))

    def test_5(self):  # ensure present dirs does not crash make_dirs
        utils.make_dirs(self.base, "bb11")
        self.assertTrue(os.path.isdir(os.path.join(self.base, "bb11")))

    def test_glob_1(self):
        f = utils.glob(os.path.join(self.base, "*"))
        f = [os.path.split(x)[1] for x in f]
        self.assertEqual(f, ["aa", "bb5", "bb11"])


class Test_isnumber(TestCase):
    def test_isnumber_int(self):
        self.assertTrue(utils.isnumber(19))

    def test_isnumber_float(self):
        self.assertTrue(utils.isnumber(19.))

    def test_isnumber_cplx(self):
        self.assertTrue(utils.isnumber(19. + 1j))


class Test(TestCase):
    def setUp(self):
        self.t0 = t0 = datetimemodule.datetime(2012, 12, 6, 8, 30, 12, 232123)

        class datetime(object):
            @classmethod
            def now(cls):
                return t0
        self.olddatetime = datetimemodule.datetime
        datetimemodule.datetime = datetime

    def tearDown(self):
        datetimemodule.datetime = self.olddatetime

    def test_1(self):
        self.assertEqual(utils.timestamp(), "20121206T083012")

    def test_2(self):
        self.assertEqual(utils.timestamp(highres=True), "20121206T083012.232")


if __name__ == '__main__':
    unittest.main()
