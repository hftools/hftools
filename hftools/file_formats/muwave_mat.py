# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

"""
Code to read data files created for the muwave matlab toolbox.
https://github.com/extrakteon/muwave

TODO:
    * When unicode is supported fully by h5py, do not convert to str

"""

import datetime
import itertools
import os
import pdb
import sys

from itertools import chain

import numpy as np
import scipy.io as sio

from numpy import array, zeros, ones, newaxis, sqrt, pi, exp, sin, cos, tan,\
    log, log10, arange, zeros_like, empty_like, linspace, angle

import hftools

from hftools import path
from hftools.dataset import hfarray, DimSweep, DimRep, DimPartial,\
    DataBlock, make_matrix
from hftools.file_formats import HFToolsIOError


def is_muwave_matlabdata(filename, rad):
    signature = "MATLAB 5.0 MAT-file"
    return rad.startswith(signature)


def build_measmnt(data):
    try:
        q = data["measmnt"][0, 0]
    except KeyError:
        q = None

    if q is None:
        return {}
    out = {}
    props = q["props"][0, 0][0]
    values = q["values"][0, 0][0]
    for (k, v) in zip(props, values):
        if v:
            out[k[0]] = hfarray(make_real(v[0]))
        else:
            out[k[0]] = hfarray("")
        if out[k[0]].dtype.name.startswith("unicode"):
            out[k[0]] = hfarray(np.char.encode(out[k[0]], "cp1252", errors="replace"), dims=out[k[0]].info)
    return out


def build_meas_state(data):
    state_names = data["measstate"][0, 0]["props"][0, 0][0]
    state_data = data["measstate"][0, 0]["values"][0, 0][0]
    d = DataBlock()
    for k, v in zip(state_names, state_data):
        if k[0] == "Index":
            d["Index"] = DimPartial("Index", v[0])
        else:
            if v[0].ndim == 1:
                d[k[0]] = hfarray(make_real(v[0][0]))
            else:
                d[k[0]] = hfarray(make_real(v[0]))
        if d[k[0]].dtype.name.startswith("unicode"):
            d[k[0]] = hfarray(np.char.encode(d[k[0]], "cp1252", errors="replace"), dims=d[k[0]].info)
    return d


def make_real(data):
    try:
        return np.array(data, np.float64)
    except:
        return data

def reorder(m):
    grouped = {}
    for dim in m.info:
        grouped.setdefault(dim.sortprio, []).append(dim)
    out = []
    for prio in sorted(grouped):
        out.extend(grouped[prio])
    return m.reorder_dimensions(*out)



def build_data(data):
    out = build_meas_state(data)
    for k, v in build_measmnt(data).iteritems():
        out[k] = hfarray(v)

    data = data["data"][0, 0]
    type = data["type"][0, 0][0]
    Z0 = float(data["reference"][0, 0][0][0])
    freq = DimSweep("freq", data["freq"][0, 0][:, 0], unit="Hz")
    arraydata = data["data"][0, 0][0]["mtrx"][0].transpose((2,0,1))
    if arraydata.dtype.name.startswith("unicode"):
        arraydata = np.char.encode(arraydata, "cp1252", errors="replace")
    if np.isrealobj(arraydata):
        arraydata = array(arraydata, np.float64)
    S = make_matrix(arraydata, (freq,))
    Scov = data["datacov"][0, 0]
    out[type] = S
    out["Z0"] = hfarray(Z0)
    if "Date" in out:
        try:
            out["Date"] = hfarray(array(datetime.datetime.strptime(out["Date"].flatten()[0], "%d-%b-%Y %H:%M:%S"), dtype="M8[us]"))
        except ValueError:
            pass
    if Scov.size:
        Scov = make_matrix(Scov[0]["mtrx"][0].transpose((2,0,1)), (freq,))
        out["%scov"%type] = Scov

    for k,v in out.vardata.iteritems():
        if k in ["V1", "V2", "V1_SET", "V2_SET"]:
            v.unit = "V"

    for k,v in out.vardata.iteritems():
        if k in ["I1", "I2", "I1_SET", "I2_SET"]:
            v.unit = "A"

    for k,v in out.vardata.iteritems():
        if k in ["Z0"]:
            v.unit = "Ohm"
    return out


def read_single_muwave_matlabdata(filename, **kw):
    data = sio.loadmat(filename)
    if "swp" not in data:
        raise MuwaveIOError("%r is not a muwave matlab file"%filename)
    swpdata = data["swp"][0, 0]["data"][0, :]
    out = []
    for d in swpdata:
        out.append(build_data(d))
    out2 = hftools.file_formats.merge_blocks(out)
    out2.replace_dim("INDEX1", DimSweep)
    return out2


def read_muwave_matlabdata(filename, drop_empty=True, **kw):
    filenames = hftools.utils.glob(filename)
    out = []
    if len(filenames) > 1:
        for idx, fname in enumerate(filenames):
            d = read_single_muwave_matlabdata(fname, drop_empty=drop_empty)
            d["Index2"] = DimPartial("Index2", [idx])
            out.append(d)
        d = hftools.file_formats.merge_blocks(out)
    else:
        d = read_single_muwave_matlabdata(filenames[0], drop_empty=drop_empty)

    order = [d.ivardata[dim] for dim in ["freq", "Index", "Index2"] if dim in d.S.info]

    d.S = d.S.reorder_dimensions(*order)
    if drop_empty:
        for k in d.vardata.keys():
            if np.all("" == d[k]):
                del d[k]
    return d


if __name__ == "__main__":
    p = path("ColdSource-muwave")
    p1 = path("ColdSource-muwave/ColdSource_Vg_m2.mat")
    p2 = path("ColdSource-muwave/ColdSource_Vg_m2_5.mat")

    d = read_muwave_matlabdata(p / "*.mat")

    figure(1)
    clf()
    subplot(221, projection="smith")
    plot(d.S11)
    subplot(224, projection="smith")
    plot(d.S22)

    figure(2)
    clf()
    subplot(111, projection="db")
    plot(d.S11)
