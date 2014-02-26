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
import unittest2 as unittest

import hftools.dataset.dim as ddim
import hftools.dataset.dataset as dset
import hftools.dataset as ds

from hftools.testing import random_value_array, random_complex_value_array,\
    random_value_array_from_dims
from hftools.dataset import hfarray, DimSweep, DimRep, DimMatrix_i,\
    DimMatrix_j, change_dim, DiagAxis
from hftools.testing import TestCase, make_load_tests
from hftools.dataset import DataDict, DataBlock
from hftools.dataset.comments import Comments

basepath = os.path.split(__file__)[0]
load_tests = make_load_tests(dset)

VA = hfarray


class Test_changedim(TestCase):
    def setUp(self):
        self.d = DataBlock()
        self.d.P = VA([0.5, -.3, .5])
        Sdims = (DimMatrix_i("i", 2), DimMatrix_j("j", 2))
        self.d.S = VA([[11, 12], [21, 22]], dims=Sdims)
        Wdims = (DimSweep("g", 1), DimMatrix_i("i", 2), DimMatrix_j("j", 2))
        self.d.W = VA([[[11, 12], [21, 22]]], dims=Wdims)

    def test_1(self):
        change_dim(self.d, DimRep, DiagAxis)
        self.assertIsInstance(self.d.ivardata["i"], DimMatrix_i)
        self.assertIsInstance(self.d.ivardata["j"], DimMatrix_j)
        self.assertIsInstance(self.d.ivardata["g"], DimSweep)

    def test_2(self):
        change_dim(self.d, DimSweep, DiagAxis)
        self.assertIsInstance(self.d.ivardata["i"], DimMatrix_i)
        self.assertIsInstance(self.d.ivardata["j"], DimMatrix_j)
        self.assertIsInstance(self.d.ivardata["g"], DiagAxis)

if __name__ == '__main__':
        d = DataBlock()
        d.comments = Comments(["Vgs=10", "Ig=10"])
        fi = DimSweep("Freq[GHz]", 3)
        gi = DimSweep("g", 4)
        d["Vds"] = hfarray([1, 2, 3], dims=(fi,))
        d["Id"] = hfarray([1, 2, 3, 4], dims=(gi,))
        d.guess_units()
