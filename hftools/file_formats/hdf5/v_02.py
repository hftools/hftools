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


def expand_dataset(dataset, expansion, axis=None):
    if axis is None:
        expansion = expansion[:]
        newsize = []
        for current, maxlen in (dataset.shape, dataset.maxshape):
            if maxlen is None:
                newsize.append(current + expansion.pop(0))
            else:
                newsize.append(current)
        dataset.resize(newsize)
    else:
        dataset.resize(dataset.shape[axis] + expansion, axis=axis)


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
    if expandable:
        if data.ndim == 0:
            maxshape = (None, )
        else:
            maxshape = list(data.shape)
            maxshape[1] = None
            maxshape = tuple(maxshape)
    else:
        maxshape = None
    db.create_dataset(key, data=data, maxshape=maxshape)
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


def save_hdf5(db, filehandle, expandable=False, expanddim=None, **kw):
    if expanddim is None:
        expanddim = DimRep("INDEX", 1)
    filehandle.attrs["hftools file version"] = "0.2"
    for k, v in db.ivardata.items():
        create_dataset(filehandle, k, v.data)
        filehandle[k].attrs["dimtype"] = v.__class__.__name__
        if v.unit:
            filehandle[k].attrs["unit"] = v.unit
        if v.outputformat:
            filehandle[k].attrs["outputformat"] = v.outputformat

    data = np.array(expanddim.data)
    filehandle.create_dataset(expanddim.name, data=data, maxshape=(None,))
    filehandle[expanddim.name].attrs["dtype"] = data.dtype.str
    filehandle[expanddim.name].attrs["dimtype"] = expanddim.__class__.__name__
    if expanddim.unit:
        filehandle[expanddim.name].attrs["unit"] = expanddim.unit
    if expanddim.outputformat:
        outputformat = expanddim.outputformat
        filehandle[expanddim.name].attrs["outputformat"] = outputformat

    for k, v in db.vardata.items():
        if expandable and v.ndim == 0:
            v = v.add_dim(expanddim)
        else:
            v = v.add_dim(expanddim, 1)
        create_dataset(filehandle, k, v, expandable=expandable)
        for idx, dv in enumerate(v.dims):
            filehandle[k].dims.create_scale(filehandle[dv.name])
            filehandle[k].dims[idx].attach_scale(filehandle[dv.name])

        filehandle[k].attrs["arraytype"] = dv.__class__.__name__
        if v.unit:
            filehandle[k].attrs["unit"] = v.unit
        if v.outputformat:
            filehandle[k].attrs["outputformat"] = v.outputformat


def append_hdf5(db, filehandle, expanddim=None, **kw):
    if filehandle.attrs["hftools file version"] != "0.2":
        raise Exception("Can only append to hftools version 0.2")
    if expanddim is None:
        expanddim = DimRep("INDEX", 1)
    expand_dataset(filehandle[expanddim.name], 1, 0)
    filehandle[expanddim.name].value[-1] = expanddim.data[0]

    for k, v in filehandle.items():
        if k in db.ivardata or k == expanddim.name:
            continue
        if k not in db.vardata:
            raise Exception("Variable %r missing in db, can not append" % k)
        diskdata = filehandle[k]
        ax = diskdata.maxshape.index(None)
        expand_dataset(diskdata, 1, ax)
        diskdata.value.take(-1, ax)[...] = db[k]
