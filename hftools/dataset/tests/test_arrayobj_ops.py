#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import operator
import os

import numpy as np

from numpy import newaxis
from hftools.testing import TestCase

import hftools.dataset.arrayobj as aobj

from hftools.testing import random_value_array_from_dims

basepath = os.path.split(__file__)[0]


class VArray(aobj.hfarray):
    __array_priority__ = 10


class Test_binary_ops(TestCase):
    op = operator.add
    randfunc = [random_value_array_from_dims]

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
        v1 = VArray(self.randfunc[0]((self.fi, self.gi, self.ri), mean=10))
        v2 = (self.randfunc[0]((self.ri,), mean=10))
        a1 = np.array(v1)
        a2 = np.array(v2)
        self._check(v1, v2, a1, a2)

    def test_2(self):
        v1 = VArray(self.randfunc[0]((self.fi, self.gi, self.ri), mean=10))
        v2 = (self.randfunc[0]((self.gi,), mean=10))
        a1 = np.array(v1)
        a2 = np.array(v2)[:, newaxis]
        self._check(v1, v2, a1, a2)

    def test_3(self):
        v1 = VArray(self.randfunc[0]((self.fi, self.ri, self.gi), mean=10))
        v2 = (self.randfunc[0]((self.ri, self.fi), mean=10))
        a1 = np.array(v1).transpose(0, 2, 1)
        a2 = np.array(v2).transpose()[:, newaxis]
        self._check(v1, v2, a1, a2)

    def test_4(self):
        v1 = (self.randfunc[0]((self.fi, self.gi, self.ri), mean=10))
        v2 = VArray(self.randfunc[0]((self.ri,), mean=10))
        a1 = np.array(v1)
        a2 = np.array(v2)
        self._check(v1, v2, a1, a2)

    def test_5(self):
        v1 = (self.randfunc[0]((self.fi, self.gi, self.ri), mean=10))
        v2 = VArray(self.randfunc[0]((self.gi,), mean=10))
        a1 = np.array(v1).transpose(1, 0, 2)
        a2 = np.array(v2)[:, newaxis, newaxis]
        self._check(v1, v2, a1, a2)

    def test_6(self):
        v1 = (self.randfunc[0]((self.fi, self.ri, self.gi), mean=10))
        v2 = VArray(self.randfunc[0]((self.ri, self.fi), mean=10))
        a1 = np.array(v1).transpose(0, 2, 1)
        a2 = np.array(v2).transpose()[:, newaxis]
        self._check(v1, v2, a1, a2)

    def test_7(self):
        v1 = VArray(self.randfunc[0]((self.fi, self.gi, self.ri), mean=10))
        v2 = 5.
        a1 = np.array(v1)

        self._check(v1, v2, a1, v2)


class Test_binary_ops_sub(Test_binary_ops):
    op = operator.sub


class Test_binary_ops_mul(Test_binary_ops):
    op = operator.mul


class Test_binary_ops_div(Test_binary_ops):
    op = operator.div


class Test_binary_ops_tdiv(Test_binary_ops):
    op = operator.truediv


class Test_binary_ops_pow(Test_binary_ops):
    op = operator.pow


def rand_bool(dims, mean):
    values = random_value_array_from_dims(dims, mean=mean)
    values = abs(values).astype(np.int32)
    return values


class Test_binary_ops_and(Test_binary_ops):
    op = operator.and_
    randfunc = [rand_bool]

    def test_7(self):
        v1 = VArray(self.randfunc[0]((self.fi, self.gi, self.ri), mean=10))
        v2 = 5
        a1 = np.array(v1)

        self._check(v1, v2, a1, v2)


class Test_binary_ops_or(Test_binary_ops_and):
    op = operator.or_


class Test_binary_ops_xor(Test_binary_ops_and):
    op = operator.xor


if __name__ == '__main__':
    fi = aobj.DimSweep("f", 3)
    gi = aobj.DimSweep("g", 4)
    ri = aobj.DimRep("r", 5)
    v1 = random_value_array_from_dims((fi, gi, ri), mean=10)
    v2 = VArray(random_value_array_from_dims((gi,), mean=10))
    a1 = np.array(v1)
    a2 = np.array(v2)[:, newaxis]
