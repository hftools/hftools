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

basepath = os.path.split(__file__)[0]

import hftools.dataset.comments as comments
import hftools.dataset.arrayobj as aobj
from hftools.dataset.comments import Comments
from hftools.testing import TestCase


class TestComment1(TestCase):
    def setUp(self):
        self.comment = Comments(["V1:10", "V2[V]:13", "V3  [V]  :  14"])

    def test_1(self):
        self.assertEqual(self.comment.property["V1"], [10])

    def test_2(self):
        self.assertEqual(self.comment.property["V2"], [13])

    def test_3(self):
        self.assertEqual(self.comment.property["V3"], [14])


class TestComment2(TestComment1):
    def setUp(self):
        self.comment = Comments(["V1=10", "V2[V]=13", "V3  [V]  =  14"])


class TestComment3(TestCase):
    def setUp(self):
        self.comment = Comments(["Vgs[V] = -1 ",
                                 "Vds[V] = 5.3",
                                 "Ig[uA] = 0.3 ",
                                 "Id[mA] = 20 ",
                                 "P = 20 mW"])

    def test_1(self):
        self.assertEqual(self.comment.property["Vgs"], [-1])

    def test_2(self):
        self.assertEqual(self.comment.property["Vds"], [5.3])

    def test_3(self):
        self.assertEqual(self.comment.property["Ig"], [0.3e-6])

    def test_4(self):
        self.assertEqual(self.comment.property["Id"], [0.02])

    def test_5(self):
        self.assertEqual(self.comment.property["P"], [0.02])


class TestDateConverter(TestCase):

    def test_conv_time_2(self):
        comments.conv_date_time("2012-12-03 12:03:12")


class Test_process_comment(TestCase):
    def test_1(self):
        self.assertEqual(comments.process_comment("lajkse"), {})

    def test_2(self):
        d = comments.process_comment("lajkse[V]:19")
        self.assertEqual(d["lajkse"], 19)
        self.assertEqual(d["lajkse"].unit, "V")


class Test_Comments(TestCase):
    def test_1(self):
        c = Comments()
        c.add_from_comment("!")
        self.assertEqual(c.fullcomments, [])

    def test_extend_1(self):
        c = Comments()
        c2 = Comments(["apa:11"])
        c.add_from_comment("!a: 10")
        self.assertEqual(c.fullcomments, ["a: 10"])
        c.extend(c2)
        self.assertEqual(c.fullcomments, ["a: 10", "apa:11"])

    def test_table1(self):
        c = Comments(["apa:10", "kul: kasdjlkj"])
        x = c.table()
        self.assertEqual(x, [' Comments '.center(77, "-"),
                             'Key                                   ' +
                             'Value                                  ',
                             '--- ' + '-' * 73,
                             'apa 10.0                              ' +
                             '                                       ',
                             'kul kasdjlkj                           ' +
                             '                                      '])

    def test_table2(self):
        c = Comments(["apa:10", "kul: kasdjlkjlo aloksd olakds okal " +
                      "sdlokasdolkaolsdola " +
                      "sdolasdlk oasdasiukweyrkh sdfsadf "])
        x = c.table()
        self.assertEqual(x, [' Comments '.center(77, "-"),
                             'Key                                   ' +
                             'Value                                  ',
                             '--- ' + '-' * 73,
                             'apa 10.0                              ' +
                             '                                       ',
                             'kul kasdjlkjlo aloksd olakds okal ' +
                             'sdlokasdolkaolsdola sdolasdlk oasdasiukweyr'])

    def test_table3(self):
        c = Comments()
        x = c.table()
        self.assertEqual(x, [])
