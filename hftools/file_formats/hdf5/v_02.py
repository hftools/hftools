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

import numpy as np
from hftools.dataset import DimRep, DimSweep, hfarray, DataBlock,\
    DimMatrix_i, DimMatrix_j, DimPartial
from hftools.py3compat import PY3

dimrep = {"DimRep": DimRep,
          "DimSweep": DimSweep,
          "DimMatrix_i": DimMatrix_i,
          "DimMatrix_j": DimMatrix_j,
          "DimPartial": DimPartial,
          }


def getvar(db, key):
    X = db[key]

    dims = []
    for x in X.dims:
        if len(x):
            dimcls = dimrep[x[0].attrs.get("dimtype", "DimSweep")]
            dim = dimcls(x[0].name.strip("/"),
                         x[0][...],
                         unit=x[0].attrs.get("unit", None))
            dims.append(dim)

    unit = X.attrs.get("unit", None)
    outputformat = X.attrs.get("outputformat", None)
    dtype = X.attrs.get("dtype", None)
    if len(dims) == len(X.dims):
        return hfarray(X.value, dims=dims, unit=unit,
                       dtype=dtype, outputformat=outputformat)
    elif len(X.dims) == 1 and "dimtype" in X.attrs:
        dimcls = dimrep[X.attrs.get("dimtype", "DimSweep")]
        dim = dimcls(X.name.strip("/"), X[...], unit=unit)
        return hfarray(X.value, dims=(dim,), dtype=dtype, unit=unit)
    else:
        return np.array(X.value, dtype=dtype)


def create_dataset(db, key, data, expandable=False):
    data_dtype = data.dtype
    if data.ndim:
        if np.issubdtype(data.dtype, np.datetime64):
            data = data.astype(np.uint64)
        elif np.issubdtype(data.dtype, np.unicode_):
            data = data.astype(h5py_string_dtype)
    else:
        if np.issubdtype(data.dtype, np.datetime64):
            data = data.astype(np.uint64)
        elif np.issubdtype(data.dtype, np.unicode_):
            data = data.astype(h5py_string_dtype)
    db.create_dataset(key, data=data)
    db[key].attrs["dtype"] = data_dtype.str


def read_hdf5(filehandle, **kw):
    db = DataBlock()
    for k in filehandle:
        db[k] = getvar(filehandle, k)
    return db

if PY3:
    h5py_string_dtype = h5py.special_dtype(vlen=str)
else:
    h5py_string_dtype = h5py.special_dtype(vlen=unicode)


def save_hdf5(db, filehandle, expandable=False, **kw):
    filehandle.attrs["hftools file version"] = "0.2"
    for k, v in db.ivardata.items():
        create_dataset(filehandle, k, v.data, False)
        filehandle[k].attrs["dimtype"] = v.__class__.__name__
        if v.unit:
            filehandle[k].attrs["unit"] = v.unit
        if v.outputformat:
            filehandle[k].attrs["outputformat"] = v.outputformat

    for k, v in db.vardata.items():
        create_dataset(filehandle, k, v, False)
        for idx, dv in enumerate(v.dims):
            filehandle[k].dims.create_scale(filehandle[dv.name])
            filehandle[k].dims[idx].attach_scale(filehandle[dv.name])

        filehandle[k].attrs["arraytype"] = dv.__class__.__name__
        if v.unit:
            filehandle[k].attrs["unit"] = v.unit
        if v.outputformat:
            filehandle[k].attrs["outputformat"] = v.outputformat


def append_hdf5(db, filehandle, **kw):
    if filehandle.attrs["hftools file version"] != "0.2":
        raise Exception("Can only append to hftools version 0.2")


