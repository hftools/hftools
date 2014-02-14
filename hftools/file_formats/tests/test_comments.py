#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import numpy as np
import os, pdb
import unittest2 as unittest
import hftools.file_formats

from unittest2 import TestCase

basepath = os.path.split(__file__)[0]

from hftools.file_formats import Comments


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
 
