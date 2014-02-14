#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import operator
import os
import pdb

import numpy as np

from numpy import newaxis
from hftools.testing import TestCase, skip, make_load_tests

import hftools.dataset.arrayobj as aobj
import hftools.dataset as ds

from hftools.testing import random_value_array, random_complex_value_array,\
    random_value_array_from_info

basepath = os.path.split(__file__)[0]


class VArray(aobj.hfarray):
    __array_priority__ = 10


class Test_binary_ops(TestCase):
    op = operator.add

    def setUp(self):
        self.fi = aobj.DimSweep("f", 3)
        self.gi = aobj.DimSweep("g", 4)
        self.ri = aobj.DimRep("r", 5)

    def _check(self, v1, v2, a1, a2):
        res_v = self.op(v1, v2)
        res_a = self.op(a1, a2)

        res_v.verify_dimension()
        self.assertAllclose(res_v, res_a)
        self.assertIsInstance(res_v, v1.__class__)

    def test_1(self):
        v1 = VArray(random_value_array_from_info((self.fi, self.gi, self.ri),
                    mean=10))
        v2 = (random_value_array_from_info((self.ri,), mean=10))
        a1 = np.array(v1)
        a2 = np.array(v2)
        self._check(v1, v2, a1, a2)

    def test_2(self):
        v1 = VArray(random_value_array_from_info((self.fi, self.gi, self.ri),
                    mean=10))
        v2 = (random_value_array_from_info((self.gi,), mean=10))
        a1 = np.array(v1)
        a2 = np.array(v2)[:, newaxis]
        self._check(v1, v2, a1, a2)

    def test_3(self):
        v1 = VArray(random_value_array_from_info((self.fi, self.ri, self.gi),
                    mean=10))
        v2 = (random_value_array_from_info((self.ri, self.fi), mean=10))
        a1 = np.array(v1).transpose(0, 2, 1)
        a2 = np.array(v2).transpose()[:, newaxis]
        self._check(v1, v2, a1, a2)

    def test_4(self):
        v1 = (random_value_array_from_info((self.fi, self.gi, self.ri),
                                           mean=10))
        v2 = VArray(random_value_array_from_info((self.ri,),
                                                 mean=10))
        a1 = np.array(v1)
        a2 = np.array(v2)
        self._check(v1, v2, a1, a2)

    def test_5(self):
        v1 = (random_value_array_from_info((self.fi, self.gi, self.ri),
                                           mean=10))
        v2 = VArray(random_value_array_from_info((self.gi,), mean=10))
        a1 = np.array(v1).transpose(1, 0, 2)
        a2 = np.array(v2)[:, newaxis, newaxis]
        self._check(v1, v2, a1, a2)

    def test_6(self):
        v1 = (random_value_array_from_info((self.fi, self.ri, self.gi),
                                           mean=10))
        v2 = VArray(random_value_array_from_info((self.ri, self.fi), mean=10))
        a1 = np.array(v1).transpose(0, 2, 1)
        a2 = np.array(v2).transpose()[:, newaxis]
        self._check(v1, v2, a1, a2)


class Test_binary_ops_sub(Test_binary_ops):
    op = operator.sub


class Test_binary_ops_mul(Test_binary_ops):
    op = operator.mul


class Test_binary_ops_div(Test_binary_ops):
    op = operator.div


class Test_binary_ops_pow(Test_binary_ops):
    op = operator.pow


if __name__ == '__main__':
    fi = aobj.DimSweep("f", 3)
    gi = aobj.DimSweep("g", 4)
    ri = aobj.DimRep("r", 5)
    v1 = random_value_array_from_info((fi, gi, ri), mean=10)
    v2 = VArray(random_value_array_from_info((gi,), mean=10))
    a1 = np.array(v1)
    a2 = np.array(v2)[:, newaxis]
