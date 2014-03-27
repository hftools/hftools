# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import itertools
from collections import OrderedDict

from numpy import array, empty
import numpy as np

from hftools.core.exceptions import HFToolsHyperCubeError
from hftools.dataset import DataDict, DimSweep, DimPartial, hfarray,\
    DataBlock, DimRep
from hftools.file_formats.common import Comments


def merge_blocks_to_association_list(blocks):
    data = DataDict()
    coords = []

    # Create DataDict with one entry per variable in the blocks
    # For each variable add a dictionary that is indexed by the DimPartials
    for b in blocks:
        coord = []
        for name, dim in b.ivardata.items():
            if isinstance(dim, DimPartial):
                coord.append((name, array(dim.data)[0]))
        coord.sort()
        coord = tuple(coord)
        coords.append(coord)
        for v in b.vardata.keys():
            data.setdefault(v, []).append((coord, b.vardata[v]))
    return data


def association_list_in_hypercube_order(association_list):
    names = [x[0] for x in association_list[0][0]]
    indices = [tuple(y[1] for y in x[0]) for x in association_list]
    uniq = [set(x) for x in zip(*indices)]
    p = set(itertools.product(*map(sorted, uniq)))

    if len(p) != len(association_list):
        msg = ("Could not make a hypercube of blocks with DimPartials.\n"
               " Indices %r are incomplete, there are %s elements but "
               "there should be %s" % (names, len(association_list), len(p)))
        raise HFToolsHyperCubeError(msg)

    out = []

    for i, int_i in zip(itertools.product(*[sorted(x) for x in uniq]),
                        itertools.product(*[range(len(x)) for x in uniq])):
        idx = indices.index(i)
        out.append((int_i, association_list[idx][1]))
    return names, map(sorted, uniq), out


def merge_variable(association_list):
    names, indices, association_list =\
        association_list_in_hypercube_order(association_list)
    innershape = tuple(len(x) for x in indices)

    innerdims = tuple(DimRep(name, i) for name, i in zip(names, indices))

    if len(association_list) == 1:
        #pdb.set_trace()
        return association_list[0][1], {}, None
    basedims = association_list[0][1].dims
    prototype = association_list[0][1]

    if prototype.dtype.type in (np.unicode_, np.str_):
        length = max(int(x[1].dtype.str[2:]) for x in association_list)
        proto_dtype = np.dtype((prototype.dtype.type, length))
    else:
        proto_dtype = prototype.dtype
    newshape = prototype.shape[:1] + innershape + prototype.shape[1:]
    result = empty(newshape, dtype=proto_dtype)
    unit = []

    for idx, x in association_list:
        unit.append(getattr(x, "unit", None))
        if prototype.ndim:
            result[(slice(None, None, None),) + idx] = x
        else:
            if x.ndim == 0:
                result[idx] = x.flatten()[0]
            else:
                result[idx] = x
    if len(set(unit)) == 1:
        unit = unit.pop()
    else:
        unit = None
    v = hfarray(result,
                dims=basedims[:1] + innerdims + basedims[1:],
                unit=unit)
    if len(association_list[0][0]) == 0:
        indexvars = {}
    else:
        indexvars = dict((x.name, hfarray(x)) for x in innerdims)
    return v, indexvars, innerdims


def merge_blocks_do_hyper(blocks):
    #import pdb;pdb.set_trace()
    outdata = DataBlock()
    data = merge_blocks_to_association_list(blocks)

    ivars = {}
    for b in blocks:
        for vname, v in b.ivardata.items():
            if vname not in ivars:
                ivars[vname] = v

    free_vars = set()
    for vname, assoc in data.items():
        free_vars.add(zip(*assoc)[0])
    free_vars = list(free_vars)
    for vname, assoc in data.items():
        v, indexvars, dim = merge_variable(assoc)
        outdata[vname] = v
        for iname, value in indexvars.items():
            outdata.ivardata[iname] = value.dims[0]

    for v in outdata.vardata.keys():
        if v in outdata.ivardata:
            del outdata.vardata[v]
    for v in outdata.vardata.values():
        for k in v.dims:
            if k.name in outdata.ivardata:
                if ((isinstance(k, DimRep) and
                     not isinstance(outdata.ivardata[k.name], DimRep))):
                    outdata.replace_dim(outdata.ivardata[k.name], k)
    cmt = Comments()
    for block in blocks:
        if block.comments:
            cmt.extend(block.comments)
    outdata.comments = cmt
    outdata.blockname = blocks[0].blockname

    for vname, v in ivars.items():
        if vname not in outdata:
            outdata[vname] = v
    #import pdb;pdb.set_trace()
    return outdata


def get_partials(db):
    partials = {}
    for k, v in db.ivardata.items():
        if isinstance(v, DimPartial):
            partials.setdefault(k, []).append(v)
    return partials


def merge_blocks(blocks, hyper=False, indexed=False):
    db = DataBlock()
    dimpartialgroups = OrderedDict()

    for b in blocks:
        parts = get_partials(b)
        partgroup = dimpartialgroups.setdefault(tuple(parts.keys()), {})
        for k, v in parts.items():
            partgroup.setdefault(k, []).extend(v)

    for idx, dims in enumerate(dimpartialgroups.values(), 1):
        for k, v in dims.items():
            dims = (DimSweep("INDEX%s" % idx, len(v)),)
            db[k] = hfarray([x.data[0] for x in v],
                            dims=dims, unit=v[0].unit)

    varnames = set()
    for b in blocks:
        for k, v in b.ivardata.items():
            if k not in db:
                db[k] = v
        for k in b.vardata.keys():
            varnames.add(k)

    for vname in varnames:
        v = []
        for b in blocks:
            if vname not in b:
                continue
            partials = get_partials(b)
            v.append(b[vname])
        if v:
            k = tuple(partials.keys())
            if k:
                ri = (db[tuple(dimpartialgroups[k].keys())[0]].dims[0],)
            else:
                ri = tuple()
            value = hfarray(v, dims=ri + v[0].dims, unit=v[0].unit)
            if v[0].dims and isinstance(v[0].dims[0], DimSweep):
                value = value.reorder_dimensions(v[0].dims[0])
            db[vname] = value
    cmt = Comments()
    for block in blocks:
        if block.comments:
            cmt.extend(block.comments)
    db.comments = cmt
    db.blockname = blocks[0].blockname

    if hyper:
        for vnames in dimpartialgroups.keys():
            if vnames:
                hyperindex = db[vnames[0]].dims[0]
                db = db.hyper(vnames, hyperindex, all=True)
    db = db.squeeze()
    return db


if __name__ == '__main__':
    blocks = []
    blocks2 = []
    an = [(1, 1, 1), (1, 1, 2),
          (2, 1, 1), (2, 1, 2),
          (2, 2, 1), (2, 2, 2),
          (2, 3, 1), (2, 3, 2)]

    a = [(1, 1, 1),
         (1, 1, 2),
         (1, 2, 1),
         (1, 2, 2),
         (2, 1, 1),
         (2, 1, 2),
         (2, 2, 1),
         (2, 2, 2),
         ]

    fi = DimSweep("f", array([10, 20, 30]))
    for i, k, j in a:
        db = DataBlock()
        db.I = DimPartial("I", array(i))
        db.K = DimPartial("K", array(k))
        db.J = DimPartial("J", array(j))
        db.x = hfarray([11, 12, 13], dims=(fi, )) * (i + k * 10 + j * 100)
        db.y = hfarray([11, 12, 13], dims=(fi, )) * (i + k * 10 + j * 100)
        blocks.append(db)

    for i, k, j in itertools.product([1, 2], [3, 4], [5, 6, 7]):
        db = DataBlock()
        db.I = DimPartial("I", array(i))
        db.K = DimPartial("K", array(k))
        db.J = DimPartial("J", array(j))
        db.x = hfarray([11, 12, 13], dims=(fi, )) * (i + k * 10 + j * 100)
        db.y = hfarray([11, 12, 13], dims=(fi, )) * (i + k * 10 + j * 100)
        blocks2.append(db)

    outdata = merge_blocks(blocks)
    outdata_hyper = merge_blocks(blocks)

    association_list = merge_blocks_to_association_list(blocks)["y"]
    p, alist, x = association_list_in_hypercube_order(association_list)
