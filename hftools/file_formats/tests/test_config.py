#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import hftools.file_formats
import hftools.file_formats.tests.base_test as base_test
from hftools.file_formats.common import Comments
from hftools import path
from hftools.testing import TestCase
import hftools.file_formats.config as hfconfig
import datetime
testpath = path(__file__).dirname()

datapath = testpath / "testdata/config"


class TestLexical(TestCase):
    def test1(self):
        cfg = hfconfig.LexicalConfig(datapath / "lexical_1.yaml")
        self.assertEqual(cfg["VNA"]["gpib"], 16)
        self.assertEqual(cfg["VNA"]["ifbw"], 100)
        self.assertEqual(cfg["VNA"]["ifbw"].unit, "Hz")
        self.assertEqual(cfg["VNA"]["timespan"],
                         datetime.timedelta(0, 72000))
        self.assertEqual(cfg["VNA"]["date"],
                         datetime.datetime(2010, 10, 10, 8, 30))
        now = datetime.datetime.now()
        tomorrow = datetime.datetime.now() + datetime.timedelta(1, 0)
        self.assertEqual(cfg["VNA"]["datelong"],
                         datetime.datetime(2010, 10, 10, 8, 30, 1))
        self.assertEqual(cfg["VNA"]["time"],
                         datetime.datetime(tomorrow.year,
                                           tomorrow.month,
                                           tomorrow.day, 0, 1, 1))
        self.assertEqual(cfg["VNA"]["time2"],
                         datetime.datetime(tomorrow.year,
                                           tomorrow.month,
                                           tomorrow.day, 0, 1, 0))
        self.assertEqual(cfg["VNA"]["time3"],
                         datetime.datetime(now.year,
                                           now.month,
                                           now.day, 23, 59, 59))
        self.assertEqual(cfg["VNA"]["testset"]["gpib"], 16)
        self.assertAllclose(cfg["VNA"]["testset"]["smooth"], [0.1, 10e-6, 10e-9])
        self.assertEqual(cfg["VNA"]["testset"]["ifbw"], 10)
        self.assertEqual(cfg["VNA"]["testset"]["ifbw"].unit, "Hz")

    def test_save(self):
        cfg = hfconfig.LexicalConfig(datapath / "lexical_1.yaml")
        cfg["no unit"] = hfconfig.SIValue("10")
        cfg.write(datapath / "slask.yaml")
        cfg = hfconfig.LexicalConfig(datapath / "slask.yaml")
        cfg.write()
