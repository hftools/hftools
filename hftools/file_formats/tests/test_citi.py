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

testpath = path(__file__).dirname()


class TestCiti_1(base_test.Test_1):
    readfun = [hftools.file_formats.read_citi]
    basepath = testpath
    dirname = "citi"
    extension = ".citi"


class TestCiti_magangle_1(base_test.Test_1):
    readfun = [hftools.file_formats.read_citi]
    basepath = testpath
    dirname = "citi"
    extension = ".citi"
    filename = "test_magangle"


class TestCiti_Comment_1(base_test.Test_Comment_1):
    readfun = [hftools.file_formats.read_citi]
    basepath = testpath
    dirname = "citi"
    extension = ".citi"


class TestCiti_Save(TestCase):
    @classmethod
    def setUpClass(cls):
        p = path(testpath / "testdata/citi/savetest")
        if not p.exists():  # pragma: no cover
            p.makedirs()

    @classmethod
    def tearDownClass(cls):
        p = path(testpath / "testdata/citi/savetest")
        p.removedirs()

    def test_save_1(self):
        d = hftools.file_formats.read_citi(testpath /
                                           "testdata/citi/test1.citi")
        fname = testpath / "testdata/citi/savetest/test1.citi"
        hftools.file_formats.save_citi(d, fname)
        e = hftools.file_formats.read_citi(fname)
        self.assertAllclose(d.S, e.S)
        fname.unlink()

    def test_save_2(self):
        d = hftools.file_formats.read_citi(testpath /
                                           "testdata/citi/test1.citi")
        d.comments = Comments(["klasjd:1"])
        d.comments.fullcomments = ["kljlk"]
        fname = testpath / "testdata/citi/savetest/test1.citi"

        hftools.file_formats.save_citi(d, fname)
        e = hftools.file_formats.read_citi(fname)
        self.assertAllclose(d.S, e.S)
        fname.unlink()

    def test_save_3(self):
        d = hftools.file_formats.read_citi(testpath /
                                           "testdata/citi/test1.citi")
        d.comments = None
        fname = testpath / "testdata/citi/savetest/test1.citi"

        hftools.file_formats.save_citi(d, fname)
        e = hftools.file_formats.read_citi(fname)
        self.assertAllclose(d.S, e.S)
        fname.unlink()

    def test_save_4(self):
        d = hftools.file_formats.read_citi(testpath /
                                           "testdata/citi/test1.citi")
        d.comments = None
        d.Gamma = d.S11
        d.mGamma = abs(d.S11)
        del d.S
        fname = testpath / "testdata/citi/savetest/test4.citi"

        hftools.file_formats.save_citi(d, fname)
        e = hftools.file_formats.read_citi(fname)

        self.assertAllclose(d.Gamma, e.Gamma)
        self.assertAllclose(abs(d.Gamma), e.mGamma)
        fname.unlink()


class TestCiti_Comment_2(base_test.Test_Comment_2):
    readfun = [hftools.file_formats.read_citi]
    basepath = testpath
    dirname = "citi"
    extension = ".citi"


class TestCiti_bad(TestCase):
    def test_bad_1(self):
        filename = testpath / "testdata/citi/bad.citi"
        self.assertRaises(hftools.file_formats.CITIFileError,
                          hftools.file_formats.read_citi, filename)

    def test_bad_2(self):
        filename = testpath / "testdata/citi/bad_2.citi"
        self.assertRaises(hftools.file_formats.CITIFileError,
                          hftools.file_formats.read_citi, filename)


class TestCiti_seg(TestCase):
    def test1(self):
        filename = testpath / "testdata/citi/dd_test_seg.citi"
        d = hftools.file_formats.read_citi(filename)
        self.assertEqual(d.S.ndim, 3)


"""
class TestCiti_2(TestCiti_1):
    def read_data(self):
        self.block = hftools.file_formats.read_citi(basepath /
                                                    "testdata/citi/test2.citi")


class TestCiti_3(TestCiti_1):
    def read_data(self):
        self.block = hftools.file_formats.read_citi(basepath /
                                                    "testdata/citi/test3.citi")

"""

