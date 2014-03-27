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

import hftools
import numpy as np
from hftools.dataset import DimRep, hfarray, DataBlock
from hftools.dataset.dim import DimBase
from hftools.file_formats.common import Comments

from hftools.py3compat import string_types, cast_unicode, cast_bytes


def unpack_dim(dim):
    return dim.__class__.__name__, dim.name, dim.unit

def save_hdf5(db, h5file, name="datablock", mode="w", compression="gzip", **kw):
    if isinstance(h5file, string_types):
        fil = h5py.File(h5file, mode)
    else:
        fil = h5file
    grp = fil.create_group(name)
    if db.blockname is None:
        blockname = "None"
    else:
        blockname = db.blockname
    grp.attrs["Blockname"] = blockname
    vardata = grp.create_group("vardata")
    ivardata = grp.create_group("ivardata")
    comments = grp.create_group("Comments")
    if db.comments:
        com = np.array([cast_bytes(x) for x in db.comments.fullcomments])
        comments.create_dataset("fullcomments", data=com, compression=compression)
#        pdb.set_trace()

    for k in db.vardata:
        datadtype = ""
        data = db[k]
        if data.ndim:
            if data.dtype.name.startswith("datetime64"):
                datadtype = data.dtype.name
                data = data.astype(np.uint64)
            dset = vardata.create_dataset(k, data=data, compression=compression, )
            klass, name, unit = zip(*[unpack_dim(dim) for dim in db[k].dims])
        else:
            if data.dtype.name.startswith("datetime64"):
                datadtype = data.dtype.name
                data = data.astype(np.uint64)
            dset = vardata.create_dataset(k, data=data) #can not compress scalars
            name = []
        dset.attrs[r"info\name"] = list(map(cast_bytes, name))
        dset.attrs[r"data\unit"] = cast_bytes(db[k].unit if db[k].unit is not None else "None")
        dset.attrs[r"data\dtype"] = datadtype

    for k in db.ivardata:
        datadtype = ""
        data = hfarray(db.ivardata[k])
        if data.dtype.name.startswith("datetime64"):
            datadtype = data.dtype.name
            data = data.astype(np.uint64)
        dset = ivardata.create_dataset(k, data=data, compression=compression)
        klass, name, unit = unpack_dim(db.ivardata[k])
        dset.attrs[r"info\class"] = klass
        dset.attrs[r"info\unit"] = cast_bytes(unit if unit is not None else "None")
        dset.attrs[r"info\dtype"] = datadtype
    if isinstance(h5file, string_types):
        fil.close()


dims_dict = dict((name, dim) for name, dim in hftools.dataset.__dict__.items() if isinstance(dim, type) and issubclass(dim, DimBase))
#print dims

def read_hdf5(h5file, name="datablock", **kw):
    if isinstance(h5file, string_types):
        fil = h5py.File(h5file, "r")
    else:
        fil = h5file
    db = DataBlock()
    grp = fil[name]
    blockname = grp.attrs["Blockname"]
    if blockname.lower() == "none":
        blockname = None
    db.blockname = blockname
    comments = grp["Comments"]

    if "fullcomments" in comments and len(comments["fullcomments"]):
        db.comments = Comments([cast_unicode(x).strip() for x in np.array(comments["fullcomments"])])
    else:
        db.comments = Comments()
    ivardata = grp["ivardata"]
    vardata = grp["vardata"]
    for k in ivardata:
        v = ivardata[k]
        datadtype = v.attrs[r"info\dtype"] or None
        dimcls = dims_dict.get(v.attrs[r"info\class"], DimRep)
        unit = str(v.attrs.get(r"info\unit", "none"))
        if unit.lower() == "none":
            unit = None
        vdata = np.array(np.array(v), dtype=datadtype)
        dim = dimcls(k, vdata, unit=unit)
        db[k] = dim
    for k in vardata:
        v = vardata[k]
        datadtype = v.attrs[r"data\dtype"] or None
        dims = tuple(db.ivardata[cast_unicode(dimname)] for dimname in v.attrs[r"info\name"])
        unit = cast_unicode(v.attrs.get(r"data\unit", "none"))
        if unit.lower() == "none":
            unit = None
        db[k] = hfarray(np.array(v), dtype=datadtype, dims=dims, unit=unit)
    if isinstance(h5file, string_types):
        fil.close()

    if kw.get("property_to_vars", False):
        db.values_from_property()
    return db

