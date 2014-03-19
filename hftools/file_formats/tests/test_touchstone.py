#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import numpy as np
import hftools.file_formats
from hftools import path
from hftools.testing import TestCase
from hftools.file_formats.tests import base_test
from hftools.file_formats import TouchstoneError, Comments
from hftools.dataset import DataBlock, hfarray, DimSweep, DimMatrix_i,\
    DimMatrix_j

testpath = path(__file__).dirname()


class TestTouchstone_1(base_test.Test_1):
    readfun = [hftools.file_formats.read_touchstone]
    basepath = testpath
    dirname = "touchstone"
    extension = ".s2p"


class TestTouchstone_1b(base_test.Test_1):
    readfun = [hftools.file_formats.read_touchstone]
    basepath = testpath
    dirname = "touchstone"
    extension = ".s2p"
    filename = "test1b"


class TestTouchstone_2(base_test.Test_2):
    readfun = [hftools.file_formats.read_touchstone]
    basepath = testpath
    dirname = "touchstone"
    extension = ".s2p"


class TestTouchstone_3(base_test.Test_3):
    readfun = [hftools.file_formats.read_touchstone]
    basepath = testpath
    dirname = "touchstone"
    extension = ".s2p"


class TestTouchstone_Comment_1(base_test.Test_Comment_1):
    readfun = [hftools.file_formats.read_touchstone]
    basepath = testpath
    dirname = "touchstone"
    extension = ".s2p"


class TestTouchstone_Comment_2(base_test.Test_Comment_2):
    readfun = [hftools.file_formats.read_touchstone]
    basepath = testpath
    dirname = "touchstone"
    extension = ".s2p"


class TestTouchstone_Noise(TestTouchstone_1):
    def read_data(self):
        filename = testpath / "testdata/touchstone/test_noise1.s2p"
        self.block = hftools.file_formats.read_touchstone(filename)
        self.facit_Fmin = 10 ** (np.array([1, 2, 3, 4]) / 10.)
        self.facit_Rn = np.array([5, 6, 7, 8]) * 50
        self.facit_Gopt = self.facit_s[:, 1, 0]
        self.comments = ["Noise"]

    def test_Fmin(self):
        self.assertTrue(np.allclose(self.facit_Fmin, self.block["Fmin"]))

    def test_Rn(self):
        self.assertTrue(np.allclose(self.facit_Rn, self.block["Rn"]))

    def test_Gopt(self):
        self.assertTrue(np.allclose(self.facit_Gopt, self.block["Gopt"]))


class TestTouchstone_Noise2(TestTouchstone_1):
    def read_data(self):
        filename = testpath / "testdata/touchstone/test_noise2.s2p"
        self.block = hftools.file_formats.read_touchstone(filename)
        self.facit_Fmin = 10 ** (np.array([1,  4]) / 10.)
        self.facit_Rn = np.array([5,  8]) * 50
        self.facit_Gopt = self.facit_s[:, 1, 0][[0, 3]]
        self.comments = ["Noise"]

    def test_Fmin(self):
        self.assertTrue(np.allclose(self.facit_Fmin, self.block["Fmin"]))

    def test_Rn(self):
        self.assertTrue(np.allclose(self.facit_Rn, self.block["Rn"]))

    def test_Gopt(self):
        self.assertTrue(np.allclose(self.facit_Gopt, self.block["Gopt"]))


class TestTouchstone_Oneport(TestTouchstone_1):
    def setUp(self):
        TestTouchstone_1.setUp(self)
        self.facit_f = np.array([0, 1, 2, 3]) * 1e9
        self.facit_s = (np.array([1, 1j, -1, -1j])[:, np.newaxis, np.newaxis])

    def read_data(self):
        filename = testpath / "testdata/touchstone/oneport_test1.s1p"
        self.block = hftools.file_formats.read_touchstone(filename)


class TestTouchstone_Fourport(TestTouchstone_1):
    def setUp(self):
        self.facit_f = np.array(np.arange(10e6, 50e9, 50e6).tolist() + [50e9])
        self.facit_s = (np.array([[[1, 2], [3, 4]]]) *
                        np.array([1, 1j, -1, -1j])[:, np.newaxis, np.newaxis])
        self.comments = ['Created Wed May 30 18:34:37 2007',
                         '4 Port Network Data from SP1.SP block',
                         'freq  reS11  imS11  reS12  imS12  reS13  imS13'
                         '  reS14  imS14',
                         'reS21  imS21  reS22  imS22  reS23  imS23'
                         '  reS24  imS24',
                         'reS31  imS31  reS32  imS32  reS33  imS33'
                         '  reS34  imS34',
                         'reS41  imS41  reS42  imS42  reS43  imS43'
                         '  reS44  imS44',
                         ''
                         ]
        self.read_data()

    def read_data(self):
        filename = testpath / "testdata/touchstone/fourport.s4p"
        self.block = hftools.file_formats.read_touchstone(filename)

    def test_size(self):
        self.assertTrue(self.block.S.shape == (1001, 4, 4))

    def test_data(self):
        pass


class TestTouchstone_errors_1(TestCase):
    def test_1(self):
        self.assertRaises(TouchstoneError,
                          hftools.file_formats.read_touchstone,
                          testpath / "testdata/touchstone/test-error_1.s2p")

    def test_2(self):
        self.assertRaises(TouchstoneError,
                          hftools.file_formats.read_touchstone,
                          testpath / "testdata/touchstone/test-error_2.s2p")

    def test_3(self):
        self.assertRaises(TouchstoneError,
                          hftools.file_formats.read_touchstone,
                          testpath / "testdata/touchstone/test-error_3.s2p")

    def test_4(self):
        self.assertRaises(TouchstoneError,
                          hftools.file_formats.read_touchstone,
                          testpath / "testdata/touchstone/test-error_4.s2p")

    def test_5(self):
        self.assertRaises(TouchstoneError,
                          hftools.file_formats.read_touchstone,
                          testpath / "testdata/touchstone/test-error_5.s2p")

    def test_6(self):
        self.assertRaises(TouchstoneError,
                          hftools.file_formats.read_touchstone,
                          testpath / "testdata/touchstone/test-error_6.s2p")

    def test_7(self):
        self.assertRaises(TouchstoneError,
                          hftools.file_formats.read_touchstone,
                          testpath / "testdata/touchstone/test-error_7.s2p")


class TestTouchstone_save(TestCase):
    def test_1(self):
        d = DataBlock()
        d.comments = Comments([])
        fi = DimSweep("freq", [0e9, 1e9, 2e9], outputformat="%15.2f")
        dims = (fi, DimMatrix_i("i", 2), DimMatrix_j("j", 2),)
        d.S = hfarray([[[1 + 1j, 1 + 2j], [2 + 1j, 2 + 2j]]] * 3,
                         dims=dims, outputformat="%.3f")
        filename = testpath / "testdata/touchstone/savetest/res_1.txt"
        hftools.file_formats.touchstone.save_touchstone(d, filename)

        resfilename = testpath / "testdata/touchstone/savetest/res_1.txt"
        resfacitname = testpath / "testdata/touchstone/savetest/facit_1.txt"
        with open(resfilename) as resfil:
            with open(resfacitname) as facitfil:
                for idx, (rad1, rad2) in enumerate(zip(resfil, facitfil)):
                    msg = ("\nFailed on line %d\n  result: %r\n  facit: %r" %
                           (idx + 1, rad1, rad2))
                    self.assertEqual(rad1, rad2, msg=msg)

    def test_2(self):
        d = DataBlock()
        d.comments = Comments(["Vg=10"])
        fi = DimSweep("freq", [0e9, 1e9, 2e9], outputformat="%15.2f")
        dims = (fi, DimMatrix_i("i", 2), DimMatrix_j("j", 2),)
        d.S = hfarray([[[1 + 1j, 1 + 2j], [2 + 1j, 2 + 2j]]] * 3,
                         dims=dims, outputformat="%.3f")
        filename = testpath / "testdata/touchstone/savetest/res_2.txt"
        hftools.file_formats.touchstone.save_touchstone(d, filename)

        resfilename = testpath / "testdata/touchstone/savetest/res_2.txt"
        facitfilename = testpath / "testdata/touchstone/savetest/facit_2.txt"
        with open(resfilename) as resfil:
            with open(facitfilename) as facitfil:
                for idx, (rad1, rad2) in enumerate(zip(resfil, facitfil)):
                    msg = ("\nFailed on line %d\n  result: %r\n  facit: %r" %
                           (idx + 1, rad1, rad2))
                    self.assertEqual(rad1, rad2, msg=msg)

