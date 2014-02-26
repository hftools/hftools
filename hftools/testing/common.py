# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import doctest
import random
import unittest2 as unittest
import warnings

from unittest2 import skip

import numpy as np
from numpy.random import randint, normal

from hftools.dataset import hfarray, DimSweep, DimRep, DimMatrix_i,\
    DimMatrix_j
from hftools.utils import HFToolsWarning, HFToolsDeprecationWarning


class TestCase(unittest.TestCase):
    def assertAllclose(self, a, b, rtol=1e-05, atol=1e-08, msg=None):
        a = np.asanyarray(a)
        b = np.asanyarray(b)
        if msg is None:
            msg = "a (%s) not close to b (%s)" % (a, b)
        self.assertTrue(np.allclose(a, b, rtol, atol), msg)

    def assertIsNotInstance(self, obj, cls):
        self.assertFalse(isinstance(obj, cls))

    def assertIsInstance(self, obj, cls):
        self.assertTrue(isinstance(obj, cls))

    def assertHFToolsWarning(self, funk, *k):
        warnings.resetwarnings()
        warnings.simplefilter("error", HFToolsWarning)
        self.assertRaises(HFToolsWarning, funk, *k)

_IDX = 1


def get_label():
    global _IDX
    label = "x_%s_" % _IDX
    _IDX += 1
    return label


def make_load_tests(module):
    """Add doctests to tests for module

    usage::
        load_tests = make_load_tests(module)
    """
    warnings.simplefilter("error", HFToolsWarning)
    warnings.simplefilter("error", HFToolsDeprecationWarning)

    def load_tests(loader, standard_tests, pattern):
        suite = unittest.TestSuite()
        suite.addTests(standard_tests)
        DocTests = doctest.DocTestSuite(module)
        suite.addTests(DocTests)
        return suite
    return load_tests

##
## Random numpy arrays


def _random_array(shape):
    return normal(size=shape)


def random_array(N, maxsize=6, minsize=1):
    shape = tuple(x.data.shape[0] for x in random_dims(N, maxsize, minsize))
    return _random_array(shape)


def random_complex_array(N, maxsize=6, minsize=1):
    shape = tuple(x.data.shape[0] for x in random_dims(N, maxsize, minsize))
    return _random_array(shape) + _random_array(shape) * 1j


def random_value_array_from_dims(dims, mean=0, scale=1):
    shape = tuple(dim.data.shape[0] for dim in dims)
    return hfarray(normal(size=shape) * scale + mean, dims)

##
## Random hfarrays


def random_value_array(N, maxsize, minsize=1):
    dims = random_dims(N, maxsize, minsize)
    return random_value_array_from_dims(dims)


def random_complex_value_array(N, maxsize, minsize=1):
    dims = random_dims(N, maxsize, minsize)
    return (random_value_array_from_dims(dims) +
            random_value_array_from_dims(dims) * 1j)


def random_dims(N, maxsize, minsize=1):
    shape = tuple(randint(minsize, maxsize, size=N))
    label = get_label()
    basedims = [DimSweep, DimRep]
    choice = random.choice
    dims = tuple(choice(basedims)(label + str(idx), x)
                 for idx, x in enumerate(shape))
    return dims


def random_complex_matrix(N, maxsize, minsize=1, Nmatrix=2, Nmatrixj=None):
    if Nmatrixj is None:  # pragma: no cover
        Nmatrixj = Nmatrix
    dims = (random_dims(N, maxsize, minsize) +
            (DimMatrix_i("i", Nmatrix), DimMatrix_j("j", Nmatrixj)))
    return (random_value_array_from_dims(dims) +
            random_value_array_from_dims(dims) * 1j)


@skip("Skipping")
def SKIP(self):  # pragma: no cover
    pass
