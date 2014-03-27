#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import hftools.file_formats
from hftools import path
from hftools.testing import TestCase
from hftools.file_formats.tests import base_test
from hftools.dataset import DataBlock, hfarray, DimSweep
from hftools.file_formats.common import Comments

testpath = path(__file__).dirname()


class TestInitSPdata_1(base_test.Test_1):
    readfun = [hftools.file_formats.read_data]
    basepath = testpath
    dirname = "sp-data"
    extension = ".txt"
    readpars = dict(verbose=False)


class TestInitSPdata_1b(base_test.Test_1):
    readfun = [hftools.file_formats.read_data]
    basepath = testpath
    dirname = "sp-data"
    extension = ".txt"
    filename = "test1b"
    readpars = dict(verbose=False)


class TestInitSPdata_multiple(TestCase):
    def test_1(self):
        filenames = [testpath / "testdata/sp-data/test1.txt"]
        hftools.file_formats.read_data(filenames, verbose=False)

    def test_2(self):
        filenames = [testpath / "testdata/sp-data/test1.txt",
                     testpath / "testdata/sp-data/test1b.txt"]
        hftools.file_formats.read_data(filenames, verbose=False)

    def test_error_1(self):
        filenames = [testpath / "testdata/sp-data/nosuchfile.txt"]
        self.assertRaises(IOError, hftools.file_formats.read_data, filenames)


class TestInitCiti_1(base_test.Test_1):
    readfun = [hftools.file_formats.read_data]
    basepath = testpath
    dirname = "citi"
    extension = ".citi"
    readpars = dict(verbose=False)

# There is no idea to make tests based on Test_2 and Test_3 as they are
# dB/arg, mag/arg


if __name__ == '__main__':
    d = DataBlock()
    d.comments = Comments(["Hej=10", "Svejs=11"])
    dims = (DimSweep("f", 3, outputformat="%.1f"),
            DimSweep("i", 2, outputformat="%.0f"),
            DimSweep("j", 2, outputformat="%.0f"), )
    d.c = hfarray([[[1, 2], [3, 4]],
                   [[10, 20], [30, 40]],
                   [[10, 20], [30, 40]]],
                  dims=dims, outputformat="%.2f")
    save = hftools.file_formats.spdata.save_spdata
    save(d, testpath / "testdata/sp-data/savetest/res_3.txt")
    d2 = hftools.file_formats.read_spdata(testpath / "testdata/sp-data/a.txt")
