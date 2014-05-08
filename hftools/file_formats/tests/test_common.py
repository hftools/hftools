#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os

import numpy as np

basepath = os.path.split(__file__)[0]

import hftools.file_formats.common as common
from hftools.testing import TestCase
from hftools.dataset import hfarray, DataBlock, DimSweep, DimMatrix_i,\
    DimMatrix_j


class TestDateFunctions(TestCase):
    def setUp(self):
        pass

    def test_conv_date(self):
        self.assertEqual(common.conv_date("2014-05-08"),
                         np.datetime64("2014-05-08"))

    def test_conv_date_error(self):
        self.assertRaises(ValueError, common.conv_date, "not a date string")

    def test_conv_datetime(self):
        self.assertEqual(common.conv_date_time("2014-05-08 08:07:06"),
                         np.datetime64("2014-05-08 08:07:06"))

    def test_conv_datetime_error(self):
        self.assertRaises(ValueError,
                          common.conv_date_time,
                          "not a date string")


class TestProcessComment(TestCase):
    def test_process_comment_1(self):
        res = common.process_comment("Kalle [mV]: 1")
        self.assertEqual(res["Kalle"], 0.001)
        self.assertEqual(res["Kalle"].unit, "V")

    def test_process_comment_2(self):
        res = common.process_comment("Kalle: 1")
        self.assertEqual(res["Kalle"], 1)
        self.assertIsNone(res["Kalle"].unit)

    def test_process_comment_3(self):
        res = common.process_comment("Kalle: ap1")
        self.assertEqual(res["Kalle"], "ap1")

    def test_process_comment_4(self):
        res = common.process_comment("Kalle ap1")
        self.assertEqual(res, {})

    def test_process_comment_5(self):
        res = common.process_comment("Kalle [mV]: ap1")
        self.assertEqual(res["Kalle"], "ap1")


class TestComments(TestCase):
    def test_table_1(self):
        c = common.Comments([])
        self.assertEqual(c.table(), [])

    def test_table_2(self):
        c = common.Comments(["Kalle [V]: 1"])
        self.assertEqual(c.table()[3:], ["Kalle 1" + " " * 70])

    def test_table_3(self):
        c = common.Comments(["Kalle [V]: a"])
        self.assertEqual(c.table()[3:], ["Kalle a" + " " * 70])


class FormatUnitHeader(TestCase):
    def test_format_unit_header_1(self):
        res = common.format_unit_header("Power", hfarray([0.1], unit="W"))
        self.assertEqual(res, "Power [W]")

    def test_format_unit_header_2(self):
        res = common.format_unit_header("Power", hfarray([0.1], unit=None))
        self.assertEqual(res, "Power")


class FormatComplexHeader(TestCase):
    def test_format_complex_header_1(self):
        res = common.format_complex_header(["Z"],
                                           [hfarray([0.1 + 1j], unit="Ohm")],
                                           "%s",
                                           "Re(%s)",
                                           "Im(%s)"
                                           )
        self.assertEqual(res, ["Re(Z) [Ohm]", "Im(Z) [Ohm]"])

    def test_format_complex_header_2(self):
        res = common.format_complex_header(["Z"],
                                           [hfarray([0.1 + 1j], unit=None)],
                                           "%s",
                                           "Re(%s)",
                                           "Im(%s)"
                                           )
        self.assertEqual(res, ["Re(Z)", "Im(Z)"])

    def test_format_complex_header_3(self):
        res = common.format_complex_header(["Z"],
                                           [hfarray([0.1], unit="Ohm")],
                                           "%s",
                                           "Re(%s)",
                                           "Im(%s)"
                                           )
        self.assertEqual(res, ["Z [Ohm]"])

    def test_format_complex_header_4(self):
        res = common.format_complex_header(["Z"],
                                           [hfarray([0.1], unit=None)],
                                           "%s",
                                           "Re(%s)",
                                           "Im(%s)"
                                           )
        self.assertEqual(res, ["Z"])

    def test_format_complex_header_5(self):
        res = common.format_complex_header(["Z"],
                                           [hfarray([0.1 + 1j], unit=None)],
                                           "%s",
                                           "%s",
                                           None
                                           )
        self.assertEqual(res, ["Z"])


class TestNormalizeNames(TestCase):
    def test_normalize_names_1(self):
        db = DataBlock()
        db.b = hfarray(1)
        db["a1/a2 raw"] = hfarray(1)
        res = common.normalize_names(db)
        self.assertTrue("a12" in res.vardata)

    def test_normalize_names_2(self):
        db = DataBlock()
        db.b = hfarray(1)
        db["Mean(A)"] = hfarray(1)
        res = common.normalize_names(db)
        self.assertTrue("A" in res.vardata)

    def test_normalize_names_3(self):
        db = DataBlock()
        db.b = hfarray(1)
        db["a1/a2 raw"] = hfarray(1)
        db["a12"] = hfarray(2)
        res = common.normalize_names(db)
        self.assertTrue("a1/a2 raw" in res.vardata)
        self.assertTrue("a12" in res.vardata)

    def test_normalize_names_error(self):
        db = DataBlock()
        db.b = hfarray(1)
        db["Mean(A)"] = hfarray(1)
        db["A"] = hfarray(2)
        self.assertRaises(ValueError, common.normalize_names, db)


class Test_make_col_from_matrix(TestCase):
    def test_make_col_from_matrix_1(self):
        header = ["S", "P"]
        dims = (DimSweep("f", 1), DimMatrix_i("i", 2), DimMatrix_j("j", 2), )
        columns = [hfarray([[[11, 12], [21, 22]]], dims=dims),
                   hfarray([10], dims=dims[:1])]
        res = common.make_col_from_matrix(header, columns, "%s%s%s")
        self.assertEqual(res, (["S11", "S12", "S21", "S22", "P"],
                               [11, 12, 21, 22, 10]))

    def test_make_col_from_matrix_2(self):
        header = ["S"]
        dims = (DimSweep("f", 1), DimMatrix_i("i", 2), DimMatrix_j("j", 2), )
        columns = [hfarray([[[11, 12], [21, 22]]], dims=dims)]
        res = common.make_col_from_matrix(header, columns, "%s%s%s",
                                          fortranorder=True)
        self.assertEqual(res, (["S11", "S21", "S12", "S22"],
                               [11, 21, 12, 22]))
