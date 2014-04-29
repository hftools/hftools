# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
try:
    import h5py
    import_failed = False
except ImportError:  # pragma: no cover
    import_failed = True
    h5py = None

from . import v_01
from . import v_02
from .helper import hdf5context

latest_version = "0.2"


if import_failed:  # pragma: no cover
    def is_hdf5(filename, rad):
        """Can not read hdf5 files, h5py packages is missing.
        Always return False.
        """
        return False

    def read_hdf5(h5file, name):
        raise NotImplementedError("Can not read hdf5 files, "
                                  "h5py packages is missing.")

    def save_hdf5(h5file, name):
        raise NotImplementedError("Can not read hdf5 files, "
                                  "h5py packages is missing.")
else:
    def is_hdf5(filename, rad):
        """Return True if filename points to a valid hdf5 file.
        """
        return h5py.is_hdf5(filename)

    def read_hdf5(h5file, name="datablock", **kw):
        with hdf5context(h5file) as fil:
            version = fil.attrs.get("hftools file version", "")
            if version == "0.2":
                return v_02.read_hdf5(fil, name=name, **kw)
            else:
                return v_01.read_hdf5(fil, name=name, **kw)

    def save_hdf5(db, filename, version=latest_version, **kw):
        if version == "0.2":
            func = v_02.save_hdf5
        else:
            func = v_01.save_hdf5
        with hdf5context(filename, mode="w") as fil:
            func(db, fil, version=version, **kw)
