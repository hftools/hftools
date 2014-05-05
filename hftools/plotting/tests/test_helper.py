# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, EngFormatter

import hftools
import hftools.utils as utils
import hftools.testing as testing
import hftools.plotting.helper as plthelp
from hftools import path
from hftools.dataset import hfarray, DataBlock, DimSweep
from hftools.testing import TestCase, make_load_tests, expectedFailure
#import warnings
#warnings.filterwarnings(action='error', category=DeprecationWarning)
#utils.reset_hftools_warnings()

#load_tests = make_load_tests(utils)
basepath = path(__file__).dirname()


class Test_db_axes(TestCase):
    proj = "db"

    def setUp(self):
        self.db = db = DataBlock()
        fi = DimSweep("freq", [1e9, 2e9, 3e9, 4e9], unit="Hz")
        db.y = hfarray([1, 2, 3, 4], dims=(fi, ), unit="V")
        db.z = hfarray([1, 2, 3, 4 + 0j], dims=(fi, ), unit="Ohm")
        self.fig = plt.figure(1)
        self.ax = plt.subplot(111, projection=self.proj)
        self.lines = plt.plot(self.db.y)
        self.outpath = p = path(basepath / "savetest")
        if not p.exists():  # pragma: no cover
            p.makedirs()

    @classmethod
    def tearDownClass(cls):
        p = path(basepath / "savetest")
        for f in p.glob("*"):  # pragma: no cover
            f.unlink()
        p.removedirs()

    def test_get_xlabel(self):
        self.ax.set_xlabel_fmt("foo []", unit="Hz")
        self.assertEqual(self.ax.get_xlabel_fmt(), "foo []")
        self.assertEqual(self.ax.get_xlabel_unit(), "Hz")
        self.ax.set_xlabel_unit("V")
        self.assertEqual(self.ax.get_xlabel_unit(), "V")

    def test_get_ylabel(self):
        self.ax.set_ylabel_fmt("bar []", unit="V")
        self.assertEqual(self.ax.get_ylabel_fmt(), "bar []")
        self.assertEqual(self.ax.get_ylabel_unit(), "V")
        self.ax.set_ylabel_unit("W")
        self.assertEqual(self.ax.get_ylabel_unit(), "W")

    def test_no_fmt(self):
        self.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        self.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        self.ax.set_xlabel_fmt("foo []", unit="Hz")
        self.ax.set_ylabel_fmt("bar []", unit="V")
        self.assertIsNone(self.ax.get_xlabel_unit())
        self.assertIsNone(self.ax.get_ylabel_unit())
        self.assertIsNone(self.ax.get_xlabel_fmt())
        self.assertIsNone(self.ax.get_ylabel_fmt())
        self.ax.set_xlabel_unit("Hz")
        self.ax.set_ylabel_unit("V")
        self.assertIsNone(self.ax.get_xlabel_unit())
        self.assertIsNone(self.ax.get_ylabel_unit())

    def test_savefig(self):
        fname = (self.outpath / "foo.png")
        plthelp.savefig(fname)
        self.assertTrue(fname.isfile())
        fname.unlink()

    def test_savefig_2(self):
        fname = (self.outpath / "foo.png")
        plthelp.savefig(fname, facetransparent=True)
        self.assertTrue(fname.isfile())
        fname.unlink()

class Test_db10_axes(Test_db_axes):
    proj = "db10"


class Test_mag_axes(Test_db_axes):
    proj = "mag"

    def test_get_xlabel(self):
        self.ax.set_xlabel_fmt("foo []", unit="Hz")
        self.assertIsNone(self.ax.get_xlabel_fmt())
        self.assertIsNone(self.ax.get_xlabel_unit())
        self.ax.set_xlabel_unit("V")
        self.assertIsNone(self.ax.get_xlabel_unit())

    def test_get_ylabel(self):
        self.ax.set_ylabel_fmt("foo []", unit="Hz")
        self.assertIsNone(self.ax.get_ylabel_fmt())
        self.assertIsNone(self.ax.get_ylabel_unit())
        self.ax.set_ylabel_unit("V")
        self.assertIsNone(self.ax.get_ylabel_unit())


class Test_magsq_axes(Test_db_axes):
    proj = "mag_square"


class Test_complex_axes(Test_db_axes):
    proj = "cplx"


class Test_groupdelay_axes(Test_db_axes):
    proj = "groupdelay"


class Test_real_axes(Test_db_axes):
    proj = "real"


class Test_imag_axes(Test_db_axes):
    proj = "imag"


class Test_deg_axes(Test_db_axes):
    proj = "deg"


class Test_unwrapdeg_axes(Test_db_axes):
    proj = "unwrapdeg"


class Test_wrapunwrappeddeg_axes(Test_db_axes):
    proj = "wrapunwrapeddeg"


class Test_rad_axes(Test_db_axes):
    proj = "rad"


class Test_unwraprad_axes(Test_db_axes):
    proj = "unwraprad"


class Test_cplxpolar_axes(Test_db_axes):
    proj = "cplxpolar"

    def test_get_xlabel(self):
        pass

    def test_get_ylabel(self):
        pass

    def test_no_fmt(self):
        pass


class Test_xsi_axes(Test_db_axes):
    proj = "x-si"

    def test_get_xlabel(self):
        self.ax.set_xlabel_fmt("foo []", unit="Hz")
        self.assertEqual(self.ax.get_xlabel_fmt(), "foo []")
        self.assertEqual(self.ax.get_xlabel_unit(), "Hz")
        self.ax.set_xlabel_unit("V")
        self.assertEqual(self.ax.get_xlabel_unit(), "V")

    def test_get_ylabel(self):
        self.ax.set_ylabel_fmt("bar []", unit="V")
        self.assertEqual(self.ax.get_ylabel_fmt(), None)
        self.assertEqual(self.ax.get_ylabel_unit(), None)
        self.ax.set_ylabel_unit("W")
        self.assertEqual(self.ax.get_ylabel_unit(), None)



class Test_top_level(TestCase):
    proj = "db"

    def setUp(self):
        self.db = db = DataBlock()
        fi = DimSweep("freq", [1e9, 2e9, 3e9, 4e9], unit="Hz")
        db.y = hfarray([1, 2, 3, 4], dims=(fi, ), unit="V")
        db.z = hfarray([1, 2, 3, 4 + 0j], dims=(fi, ), unit="Ohm")
        self.fig = plt.figure(1)
        self.ax = plt.subplot(111, projection=self.proj)
        self.lines = plt.plot(self.db.y)
        self.outpath = p = path(basepath / "savetest")
        if not p.exists():  # pragma: no cover
            p.makedirs()

    @classmethod
    def tearDownClass(cls):
        p = path(basepath / "savetest")
        for f in p.glob("*"):
            f.unlink()
        p.removedirs()

    def test_xlabel_fmt_1(self):
        plthelp.xlabel_fmt("foo []", unit="A")
        self.assertEqual(self.ax.get_xlabel_fmt(), "foo []")
        self.assertEqual(self.ax.get_xlabel_unit(), "A")
        self.ax.set_xlabel_unit("V")
        self.assertEqual(self.ax.get_xlabel_unit(), "V")

    def test_xlabel_fmt_2(self):
        plthelp.xlabel_fmt("foo []", unit="A", axes=self.ax)
        self.assertEqual(self.ax.get_xlabel_fmt(), "foo []")
        self.assertEqual(self.ax.get_xlabel_unit(), "A")
        self.ax.set_xlabel_unit("V")
        self.assertEqual(self.ax.get_xlabel_unit(), "V")

    def test_xlabel_fmt_3(self):
        plthelp.xlabel_fmt("foo []", axes=self.ax)
        self.assertEqual(self.ax.get_xlabel_fmt(), "foo []")
        self.assertEqual(self.ax.get_xlabel_unit(), "Hz")

    def test_ylabel_fmt_1(self):
        plthelp.ylabel_fmt("foo []", unit="A")
        self.assertEqual(self.ax.get_ylabel_fmt(), "foo []")
        self.assertEqual(self.ax.get_ylabel_unit(), "A")
        self.ax.set_ylabel_unit("V")
        self.assertEqual(self.ax.get_ylabel_unit(), "V")

    def test_ylabel_fmt_2(self):
        plthelp.ylabel_fmt("foo []", unit="A", axes=self.ax)
        self.assertEqual(self.ax.get_ylabel_fmt(), "foo []")
        self.assertEqual(self.ax.get_ylabel_unit(), "A")
        self.ax.set_ylabel_unit("V")
        self.assertEqual(self.ax.get_ylabel_unit(), "V")

    def test_ylabel_fmt_3(self):
        plthelp.ylabel_fmt("foo []", axes=self.ax)
        self.assertEqual(self.ax.get_ylabel_fmt(), "foo []")
        self.assertEqual(self.ax.get_ylabel_unit(), "V")

    def test_save_all_to_pdf(self):
        fname = (self.outpath / "foo.pdf")
        plthelp.save_all_figures_to_pdf(fname)
        self.assertTrue(fname.isfile())


    def test_save_all_to_pdf_err(self):
        self.assertRaises(ValueError, plthelp.save_all_figures_to_pdf,
                          self.outpath / "foo.png")


class Test_projs(TestCase):
    def test_none(self):
        a = [1, 2, 3]
        self.assertEqual(plthelp._projfun[None](a, a), (a, a))

    def test_db(self):
        a = np.array([1, 2, 3])
        x, y = plthelp._projfun["db"](a, a)
        self.assertAllclose(y, hftools.math.dB(a))
        self.assertAllclose(a, x)
