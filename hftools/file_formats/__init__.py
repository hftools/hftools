# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
# pylint: disable=unused-import
# pyflake: disable=W0611


"""
hftools.file_formats
====================

This package contains routines to read and write data.

.. autofunction:: hftools.file_formats.read_data

.. automodule:: hftools.file_formats.mdif
.. automodule:: hftools.file_formats.touchstone
.. automodule:: hftools.file_formats.spdata
.. automodule:: hftools.file_formats.citi

"""
import glob
import re
from contextlib import contextmanager
from hftools.core.exceptions import HFToolsIOError
from hftools.file_formats.common import Comments, normalize_names
from hftools.file_formats.merge import merge_blocks
from hftools.file_formats.mdif import read_mdif, save_mdif, is_mdif
from hftools.file_formats.touchstone import read_touchstone, save_touchstone,\
    is_touchstone, TouchstoneError
from hftools.file_formats.citi import read_citi, save_citi, is_citi,\
    CITIFileError
from hftools.file_formats.spdata import read_spdata, save_spdata
from hftools.file_formats.hdf5 import read_hdf5, save_hdf5, is_hdf5,\
    append_hdf5
from hftools.file_formats.muwave_mat import read_muwave_matlabdata,\
    is_muwave_matlabdata
from hftools._external import path

isfile = [(is_mdif, read_mdif),
          (is_hdf5, read_hdf5),
          (is_citi, read_citi),
          (is_touchstone, read_touchstone),
          (is_muwave_matlabdata, read_muwave_matlabdata),
          ]

glob_pat = re.compile("[*?]")


def read_data(filename, merge=True, guess_unit=True,
              property_to_vars=True, **kw):
    """Guess fileformat of *filename* and read data.

    The file format is guessed by looking for distinguishing marks for
    citi, touchstone, and mdif. If non of those are present spdata is assumed.
    """
    if isinstance(filename, (list, tuple)):
        filenames = filename
    else:
        filenames = [filename]
    for f in filenames:
        files = glob.glob(f)
        if len(files) != 0:
            fname = files[0]
            break
    else:
        raise IOError("No files match pattern %r" % filename)
    fil = open(fname)
    for rad in fil:
        for fun, readfun in isfile:
            if fun(fname, rad):
                fil.close()
                return readfun(filename, merge=merge, guess_unit=guess_unit,
                               property_to_vars=property_to_vars, **kw)
    fil.close()
    return read_spdata(filename, merge=merge, guess_unit=guess_unit,
                       property_to_vars=property_to_vars, **kw)


@contextmanager
def read_to_cache(filename, cachename=None, cachedir=None,
                  reread=False, verbose=True):
    if cachedir is None:
        cachedir = path.getcwd() / "_hfcache"
    else:
        cachedir = path(cachedir)
    filename = path(filename)
    cachedir.makedirs(silent=True)
    if cachename is None:
        namebase = filename.namebase
    else:
        namebase = cachename
    cachename = cachedir / namebase + ".hdf5"
    if not cachename.exists() or reread:
        data = read_data(filename, verbose=verbose)
        yield data, True
        save_hdf5(data, cachename)
    else:
        data = read_hdf5(cachename)
        yield data, False


def read_from_cache(cachename=None, cachedir=None, reread=False):
    if cachedir is None:
        cachedir = path.getcwd() / "_hfcache"
    else:
        cachedir = path(cachedir)
    cachename = cachedir / cachename + ".hdf5"
    return read_hdf5(cachename)

