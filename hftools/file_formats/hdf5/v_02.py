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
from hftools.dataset.comments import Comments
#from hftools.file_formats.hdf5.hdf5 import hdf5context
from .helper import hdf5context

dimrep = {"DimRep": DimRep,
          "DimSweep": DimSweep,
          "DimMatrix_i": DimMatrix_i,
          "DimMatrix_j": DimMatrix_j,
          "DimPartial": DimPartial,
          }


def expand_dataset(dataset, expansion, axis):
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
    else:
        dimcls = dimrep[X.attrs.get("dimtype", "DimSweep")]
        dim = dimcls(X.name.strip("/"), X[...], unit=unit)
        return hfarray(X.value, dims=(dim,), dtype=dtype, unit=unit)


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
        if data.ndim in (0, 1):
            maxshape = (None, )
        else:
            maxshape = list(data.shape)
            maxshape[1] = None
            maxshape = tuple(maxshape)
    else:
        maxshape = None
    db.create_dataset(key, data=data, maxshape=maxshape)
    db[key].attrs["dtype"] = data_dtype.str


def read_hdf5_handle(filehandle, **kw):
    if isinstance(filehandle, h5py.File):
        db = DataBlock()
        for k in filehandle:
            db[k] = getvar(filehandle, k)
        db.comments = Comments()
        return db
    else:
        raise IOError("filehandle should be a h5py File, is: %r" % filehandle)


def read_hdf5(h5file, name="datablock", **kw):
    with hdf5context(h5file) as fil:
        d = read_hdf5_handle(fil, name=name, **kw)
    return d


if PY3:
    h5py_string_dtype = h5py.special_dtype(vlen=str)
else:  # if PY2
    h5py_string_dtype = h5py.special_dtype(vlen=unicode)


def save_hdf5_handle(db, filehandle, expandable=False, expanddim=None, **kw):
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

    if expandable:
        data = np.array(expanddim.data)
        filehandle.create_dataset(expanddim.name, data=data, maxshape=(None,))
        filehandle[expanddim.name].attrs["dtype"] = data.dtype.str
        expanddim_class = expanddim.__class__.__name__
        filehandle[expanddim.name].attrs["dimtype"] = expanddim_class
        if expanddim.unit:
            filehandle[expanddim.name].attrs["unit"] = expanddim.unit
        if expanddim.outputformat:
            outputformat = expanddim.outputformat
            filehandle[expanddim.name].attrs["outputformat"] = outputformat

    for k, v in db.vardata.items():
        if expandable and v.ndim == 0:
            v = v.add_dim(expanddim)
        elif expandable:
            v = v.add_dim(expanddim, 1)
        else:
            pass
        create_dataset(filehandle, k, v, expandable=expandable)
        for idx, dv in enumerate(v.dims):
            filehandle[k].dims.create_scale(filehandle[dv.name])
            filehandle[k].dims[idx].attach_scale(filehandle[dv.name])

        filehandle[k].attrs["arraytype"] = v.__class__.__name__
        if v.unit:
            filehandle[k].attrs["unit"] = v.unit
        if v.outputformat:
            filehandle[k].attrs["outputformat"] = v.outputformat


def save_hdf5(db, h5file, expandable=False, expanddim=None, **kw):
    with hdf5context(h5file, mode="w") as fil:
        d = save_hdf5_handle(db, fil,
                             expandable=expandable,
                             expanddim=expanddim,
                             **kw)
    return d


def append_hdf5(db, filehandle, expanddim=None, **kw):
    key = "hftools file version"
    if key not in filehandle.attrs or filehandle.attrs[key] != "0.2":
        raise Exception("Can only append to hftools version 0.2")
    if expanddim is None:
        expanddim = DimRep("INDEX", 1)
    idx = filehandle[expanddim.name][-1] + 1
    expand_dataset(filehandle[expanddim.name], 1, 0)
    filehandle[expanddim.name].value[-1] = idx

    for k, v in filehandle.items():
        if k in db.ivardata or k == expanddim.name:
            continue
        if k not in db.vardata:
            raise ValueError("Variable %r missing in db, can not append" % k)
        diskdata = filehandle[k]
        ax = diskdata.maxshape.index(None)
        expand_dataset(diskdata, 1, ax)
        s = (slice(None),) * ax + (diskdata.shape[ax] - 1,)
        diskdata[s] = db[k]
