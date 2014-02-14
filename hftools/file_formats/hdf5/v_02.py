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
except ImportError: # pragma: no cover
    import_failed = True
    h5py = None

import numpy as np
from hftools.dataset import DimRep, DimSweep, hfarray, DataBlock,\
    DimMatrix_i, DimMatrix_j


dimrep = {"DimRep": DimRep,
          "DimSweep": DimSweep,
          "DimMatrix_i": DimMatrix_i,
          "DimMatrix_j": DimMatrix_j,}


def getvar(db, key):
    X = db[key]
    

    info = []
    for x in X.dims:
        if len(x):
            dim = dimrep[x[0].attrs.get("dimtype", "DimSweep")](x[0].name.strip("/"), x[0][...], unit=x[0].attrs.get("unit", None))
            info.append(dim)

    unit = X.attrs.get("unit", None)
    outputformat = X.attrs.get("outputformat", None)
    if len(info) == len(X.dims):
        return hfarray(X, dims=info, unit=unit, outputformat=outputformat)
    elif len(X.dims) == 1 and "dimtype" in X.attrs:
        dim = dimrep[X.attrs.get("dimtype", "DimSweep")](X.name.strip("/"), X[...], unit=unit)
        return hfarray(X, dims=(dim,), unit=unit)
    else:
        return np.array(X)

def read_hdf5(filehandle, **kw):
    db = DataBlock()
    for k in filehandle:
        db[k] = getvar(filehandle, k)
    return db

def save_hdf5(db, filehandle, **kw):
    filehandle.attrs["hftools file version"] = "0.2"
    for k, v in db.ivardata.items():
        data = v.data
        if data.ndim:
            if data.dtype.name.startswith("datetime64"):
                datadtype = data.dtype.name
                data = data.astype(np.uint64)
            dset = filehandle.create_dataset(k, data=data)
        else:
            if data.dtype.name.startswith("datetime64"):
                datadtype = data.dtype.name
                data = data.astype(np.uint64)
            dset = filehandle.create_dataset(k, data=data)
        filehandle[k].attrs["dimtype"] = v.__class__.__name__
        if v.unit:
            filehandle[k].attrs["unit"] = v.unit
        if v.outputformat:
            filehandle[k].attrs["outputformat"] = v.outputformat
        
    for k, data in db.vardata.items():
        if data.ndim:
            if data.dtype.name.startswith("datetime64"):
                datadtype = data.dtype.name
                data = data.astype(np.uint64)
            dset = filehandle.create_dataset(k, data=data)
        else:
            if data.dtype.name.startswith("datetime64"):
                datadtype = data.dtype.name
                data = data.astype(np.uint64)
            dset = filehandle.create_dataset(k, data=data)
        filehandle[k].attrs["arraytype"] = data.__class__.__name__
        if data.unit:
            filehandle[k].attrs["unit"] = data.unit
        if data.outputformat:
            filehandle[k].attrs["outputformat"] = data.outputformat
        for idx, dv in enumerate(data.info):
            filehandle[k].dims.create_scale(filehandle[dv.name])
            filehandle[k].dims[idx].attach_scale(filehandle[dv.name])  
