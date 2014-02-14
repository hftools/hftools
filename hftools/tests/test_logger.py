#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os
import pdb
import unittest2 as unittest

import numpy as np

import hftools.math as hfmath
from hftools.dataset import hfarray, DimSweep, DimRep, DimMatrix_i,\
    DimMatrix_j
from hftools.testing import TestCase
import hftools.logger as logger

basepath = os.path.split(__file__)[0]


class Test_log(TestCase):
    def test_startlog(self):
        "Check if we can call start_log without exception"
        logger.start_log()
        logger.start_log()
        self.assertTrue(True)

    def test_stop_log(self):
        "Check if we can call start_log without exception"
        logger.stop_log()
        logger.stop_log()
        #Just make sure the nullhandler don't crash
        logger.logger.warn("Foo")
        self.assertTrue(True)

    def test_stop_log_2(self):
        "Check if we can call start_log without exception"
        logger.stop_log()
        logger.start_log()
        logger.start_log()
        logger.stop_log()
        #Just make sure the nullhandler don't crash
        logger.logger.warn("Foo")
        self.assertTrue(True)

    def test_warn(self):
        logger.logger.warn("Foo")
        
