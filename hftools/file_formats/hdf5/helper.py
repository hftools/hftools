# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
from contextlib import contextmanager
try:
    import h5py
    import_failed = False
except ImportError:  # pragma: no cover
    import_failed = True
    h5py = None


@contextmanager
def hdf5context(filehandle, mode="r"):
    if isinstance(filehandle, h5py.File):
        yield filehandle
    else:
        with h5py.File(filehandle, mode=mode) as fil:
            yield fil

