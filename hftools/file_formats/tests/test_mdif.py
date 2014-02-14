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
import hftools.file_formats.tests.base_test as base_test
from hftools.dataset import DataBlock, hfarray, DimSweep
from hftools.file_formats.common import Comments
from hftools.file_formats import read_mdif, save_mdif
import hftools.file_formats.mdif as mdif

testpath = path(__file__).dirname()


class Test_parse_header(TestCase):
    def test_1(self):
        D = mdif.GetData([["X(real)"]])
        q = D.parse_header("X(real)")
        self.assertEqual(q, (D.get_real, "X"))

    def test_2(self):
        D = mdif.GetData([["X(real)"]])
        self.assertRaises(Exception, D.parse_header, "X(unknown)")


class TestMDIFdata_1(base_test.Test_1):
    readfun = [hftools.file_formats.read_mdif]
    basepath = testpath
    dirname = "mdif"
    extension = ".mdif"
    readpars = dict(blockname="Spar", verbose=False)

# There is no point to make tests based on Test_2 and
#  Test_3 as they are dB/arg, mag/arg


class TestMDIFdata_Comment_1(base_test.Test_Comment_1):
    readfun = [hftools.file_formats.read_mdif]
    basepath = testpath
    dirname = "mdif"
    extension = ".mdif"
    readpars = base_test.Test_Comment_1.readpars.copy()
    readpars.update(dict(blockname="Spar", verbose=False))


class TestMDIFdata_Comment_2(base_test.Test_Comment_2):
    readfun = [hftools.file_formats.read_mdif]
    basepath = testpath
    dirname = "mdif"
    extension = ".mdif"
    readpars = base_test.Test_Comment_2.readpars.copy()
    readpars.update(dict(blockname="Spar", verbose=False))


class TestMDIF_readfile(TestCase):
    def test_1(self):
        mdif = read_mdif(testpath / "testdata/mdif/small.mdif", verbose=False)
        self.assertTrue("DSCR(RFmeas.LDMOS69.Spar)" in mdif)
        self.assertTrue("RFmeas.LDMOS69.Spar" in mdif)

    def test_attrib(self):
        mdif = read_mdif(testpath / "testdata/mdif/test-attrib.mdif",
                         verbose=False, blockname="Spar")
        self.assertTrue("Power" in mdif)


class TestMDIF_savefile(TestCase):
    def test_1(self):
        mdif = read_mdif(testpath / "testdata/mdif/small.mdif", verbose=False)
        save_mdif(mdif['RFmeas.LDMOS69.Spar'], testpath / "temp.mdif")

if __name__ == '__main__':
    d = DataBlock()
    d.comments = Comments(["Hej=10", "Svejs=11"])
    info = (DimSweep("f", 3, outputformat="%.1f"),
            DimSweep("i", 2, outputformat="%.0f"),
            DimSweep("j", 2, outputformat="%.0f"), )
    d.c = hfarray([[[1, 2], [3, 4]],
                      [[10, 20], [30, 40]],
                      [[10, 20], [30, 40]]], dims=info, outputformat="%.2f")
    filename = testpath / "testdata/sp-data/savetest/res_3.txt"
    hftools.file_formats.spdata.save_spdata(d, filename)
    d2 = hftools.file_formats.read_spdata(testpath / "testdata/sp-data/a.txt")
