# -*- coding: utf-8 -*
from __future__ import print_function
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

u"""
dataset
========


TODO:
    * Fixa rename av ivardata.

"""

import re

import numpy as np
import numpy.random as rnd
from numpy import zeros, array, linspace

from hftools.dataset.arrayobj import hfarray, ismatrix,\
    remove_rep, _hfarray
from hftools.dataset.dim import DimBase, DimSweep, DimRep,\
    DimMatrix_i, DimMatrix_j, DiagAxis
from hftools.utils import warn, stable_uniq
from hftools.dataset.helper import guess_unit_from_varname
from hftools.py3compat import cast_unicode, cast_str


class DataBlockError(Exception):
    pass


def subset_datablock_by_dims(db, dims):
    """Get subset of db with variables with matching dims."""
    out = DataBlock()
    dimset = set(dims)
    if len(dims) != len(dimset):
        msg = "dims %r did not specify unique dimensions %r" % (dims, dimset)
        raise ValueError(msg)
    for k, v in db.vardata.items():
        if set(v.dims) == dimset:
            out[k] = v
        elif set(v.dims) == set(dims[1:]):
            out[k] = v
    return out


def yield_dim_consistent_datablocks(db):
    dimset = set()
    for v in db.vardata.values():
        dimset.add(v.dims)
    while dimset:
        longest = sorted(dimset, key=len)[-1]
        yield longest, subset_datablock_by_dims(db, longest)
        dimset.discard(longest)
        if len(longest) > 1:
            dimset.discard(longest[1:])


def convert_matrices_to_elements(db, formatelement=None):
    out = DataBlock()
    out.blockname = db.blockname
    if formatelement is None:
        def formatelement(varname, i, j):
            return "%s%s%s" % (varname, i, j)

    for k, v in db.vardata.items():
        if ismatrix(v):
            for i, _ in enumerate(v.dims[v.dims_index("i")].data, 1):
                for j, _ in enumerate(v.dims[v.dims_index("j")].data, 1):
                    out[formatelement(k, i, j)] = v[..., i - 1, j - 1]
        else:
            out[k] = v
    return out


class DataDict(dict):
    def __init__(self, *k, **kw):
        dict.__init__(self, *k, **kw)
        self.__dict__["order"] = []

    @property
    def outputformat(self):
        for x in self.values():
            if hasattr(x, "outputformat"):
                return x.outputformat
        return "%.16e"

    @outputformat.setter
    def outputformat(self, value):
        for name in self:
            self[name].outputformat = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(exc)

    def __delattr__(self, key):
        self.__delitem__(key)

    def __delitem__(self, key):
        if key in self:
            dict.__delitem__(self, key)
        if key in self.order:
            del self.order[self.order.index(key)]

    def __setattr__(self, key, value):
        try:
            object.__getattribute__(self, key)
            object.__setattr__(self, key, value)
            return
        except AttributeError:
            pass
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        if key not in self.order:
            self.order.append(key)

    def setdefault(self, key, value):
        a = dict.setdefault(self, key, value)
        if key not in self.order:
            self.order.append(key)
        return a

    def rename(self, oldname, newname):
        vdata = self[oldname]
        if newname not in self.order:
            self.order[self.order.index(oldname)] = newname
        del self[oldname]
        self[newname] = vdata

    def view(self):
        out = self.__class__([(k, v.view()) for k, v in self.iteritems()])
        out.order = self.order[:]
        return out

    def copy(self):
        out = self.__class__([(k, v.copy()) for k, v in self.iteritems()])
        out.order = self.order[:]
        return out

    def keys(self):
        return [v for v in self]

    def iteritems(self):
        for v in self:
            yield v, self[v]

    def values(self):
        return [self[v] for v in self]

    def items(self):
        return [(v, self[v]) for v in self]

    def __iter__(self):
        out = self.order[:]
        for k in dict.keys(self):
            if k not in out:
                out.append(k)
        for k in out:
            yield k

    def __dir__(self):  # pragma: no cover
        return self.order + dir(type(self)) + self.__dict__.keys()

reg_matrix_name = re.compile("([_A-Za-z]([_A-Za-z0-9]*"
                             "[_A-Za-z])?)([0-9][0-9])")


def is_matrix_name(x):
    res = reg_matrix_name.match(x)
    if res:
        return res.groups()[0], tuple(int(x) - 1 for x in res.groups()[2])
    else:
        return None, None


def change_dim(ds, olddimclass=DimSweep, newdimclass=DiagAxis):
    for x in ds.ivardata.values():
        if isinstance(x, olddimclass):
            ds.replace_dim(x, newdimclass(x))
    return ds


class DataBlock(object):
    def __init__(self):
        self.__dict__["_blockname"] = None
        self.__dict__["comments"] = None

        self.__dict__["vardata"] = DataDict()

        #independent data
        self.__dict__["ivardata"] = DataDict()
        self.__dict__["_outputformat"] = "%.16e"
        self.__dict__["_xname"] = None

    def replace_dim(self, olddim, newdim):
        if isinstance(olddim, (str, unicode)):
            if olddim not in self.ivardata:
                msg = "%r dimension not present in datablock ivars: %r"
                msg = msg % (olddim, self.ivardata.keys())
                raise KeyError(msg)
            else:
                olddim = self.ivardata[olddim]
        if olddim.name not in self.ivardata:
            msg = "%r dimension not present in datablock ivars: %r"
            msg = msg % (olddim, self.ivardata.keys())
            raise KeyError(msg)
        if np.issubclass_(newdim, DimBase):
            newdim = newdim(olddim)
        self.ivardata.rename(olddim.name, newdim.name)
        self.ivardata[newdim.name] = newdim
        for v in self.vardata.values():
            v.replace_dim(olddim, newdim)

    def rename(self, oldname, newname):
        if oldname in self.ivardata:
            olddim = self.ivardata[oldname]
            newdim = olddim.__class__(olddim, name=newname)
            self.replace_dim(olddim, newdim)
        elif oldname in self.vardata:
            self.vardata.rename(oldname, newname)
        else:
            pass

    def values_from_property(self):
        """Convert numeric properties to values"""
        if self.comments:
            for k, v in self.comments.property.iteritems():
                if hasattr(v, "dtype") or isinstance(v, (int, float)):
                    if (k not in self.vardata) and (k not in self.ivardata):
                        self[k] = hfarray(v)

    @property
    def blockname(self):
        if self._blockname is None:
            if "FILENAME" in self:
                return cast_unicode(self.FILENAME.flat.next())
        return self._blockname

    @blockname.setter
    def blockname(self, value):
        self._blockname = value

    @property
    def xname(self):
        return self._xname

    @xname.setter
    def xname(self, name):
        self._xname = name

    @property
    def xvalue(self):
        return self[self._xname]

    def __getitem__(self, key):
        try:
            return hfarray(self.ivardata[key])
        except KeyError:
            try:
                return self.vardata[key]
            except KeyError:
                pass
        mname, elem = is_matrix_name(key)
        if mname:
            data = self.vardata[mname]
            if ismatrix(data):
                idx = (Ellipsis,) + elem
                return data[idx]
            else:
                msg = "No matrix with name %r is available"
                msg = msg % mname
                raise KeyError(msg)
        else:
            raise KeyError("No data named %r is available" % (key,))

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError as exc:
            raise AttributeError(exc)

    def __delitem__(self, key):
        if key in self.vardata:
            del self.vardata[key]

    def __delattr__(self, key):
        del self[key]

    def __setitem__(self, key, value):
        if isinstance(value, DimBase):
            self.ivardata[key] = value
            if self._xname is None:
                self._xname = value.name
        else:
            if self._xname is None and value.dims:
                self._xname = value.dims[0].name
            else:
                self._xname = None
            mname, elem = is_matrix_name(key)
            if mname in self.vardata and not ismatrix(value):
                idx = (Ellipsis,) + elem
                self.vardata[mname][idx] = value
            else:
                if (key in self.ivardata):
                    olddim = self.ivardata[key]
                    if len(value.dims) != 1:
                        msg = "ndim mismatch when trying to set ivardata,"\
                              " can only have one dimension"
                        raise AttributeError(msg)
                    self.replace_dim(olddim, olddim.__class__(olddim,
                                                              data=value))
                else:
                    self.vardata[key] = value
                for dim in value.dims:
                    if dim.name not in self.ivardata:
                        self.ivardata[dim.name] = dim

    def __setattr__(self, key, value):
        try:
            object.__getattribute__(self, key)
            object.__setattr__(self, key, value)
            return
        except AttributeError:
            pass
        self.__setitem__(key, value)

    def __contains__(self, key):
        return (key in self.vardata) or (key in self.ivardata)

    def set_outputformat(self, name, format):
        if name in self.vardata:
            self[name].outputformat = format
        elif name in self.ivardata:
            olddim = self.ivardata[name]
            newdim = olddim.__class__(olddim, outputformat=format)
            self.replace_dim(olddim, newdim)
        else:
            msg = "%r not value or index in %r"
            msg = msg % (name, self.blockname)
            raise DataBlockError(msg)

    @property
    def outputformat(self):
        return self._outputformat

    @outputformat.setter
    def outputformat(self, value):
        self._outputformat = value
        self.vardata.outputformat = value
        for olddim in self.ivardata.values():
            newdim = olddim.__class__(olddim, outputformat=value)
            self.replace_dim(olddim, newdim)

    @property
    def allvarnames(self):
        out = list(self.ivardata.keys())
        for x in self.vardata.keys():
            if x not in out:
                out.append(x)
        return out

    def copy(self):
        out = DataBlock()
        out.blockname = self.blockname
        if self.comments:
            out.comments = self.comments.copy()
        out.vardata = self.vardata.copy()
        out.ivardata = self.ivardata.copy()
        out.xname = self.xname
        return out

    def view(self):
        out = DataBlock()
        out.blockname = self.blockname
        out.comments = self.comments
        out.vardata = self.vardata.view()
        out.ivardata = self.ivardata.copy()
        out.xname = self.xname
        return out

    def filter(self, boolarray):
        if boolarray.squeeze().ndim > 1:
            raise ValueError("filter can only use array with one dimension")
        if boolarray.dtype != np.dtype(bool):
            if boolarray.dims[0].name not in self.ivardata:
                out = self.copy()
                msg = "Filter warning: DataBlock does not contain dimension %r"
                msg = msg % boolarray.dims[0]
                warn(msg)
                return out
            localdim = self.ivardata[boolarray.dims[0].name]
            intersection = set(localdim.data).intersection(boolarray)
            barray = boolarray.astype(bool)
            barray[...] = False
            for f in intersection:
                barray[boolarray == f] = True
            boolarray = barray

        out = DataBlock()
        out.blockname = self.blockname
        out.comments = self.comments
        out.xname = self.xname

        olddim = boolarray.dims[0]
        newdim = olddim.__class__(olddim, data=olddim.data[boolarray])

        for v in self.vardata.keys():
            data = self.vardata[v].view()
            try:
                i = data.dims_index(boolarray.dims[0])
            except IndexError:  # This variable does not sweep in boolarray dim
                out[v] = data
                continue
            newinfo = list(data.dims)
            reorder = range(data.ndim)
            del newinfo[i]
            del reorder[i]
            reorder = tuple([i] + reorder)
            newinfo = tuple([newdim] + newinfo)
            data = data.transpose(*reorder)
            utdata = hfarray(array(data)[boolarray], dims=newinfo,
                             unit=data.unit,
                             outputformat=data.outputformat)
            dimorder = [x.name for x in self.vardata[v].dims]
            utdata = utdata.reorder_dimensions(*dimorder)
            out[v] = utdata
        out.xname = self.xname
        return out

    def sort(self, dim):
        dim = hfarray(dim, copy=False).squeeze()
        if dim.ndim > 1:
            msg = "sort can only get sort direction from array with"\
                  " one dimension"
            raise ValueError(msg)
        else:
            dimname = dim.dims[0].name

        if dimname not in self.ivardata:
            msg = "Sort warning: DataBlock does not contain dimension %r"
            msg = msg % dimname
            warn(msg)
            out = self.copy()
            return out

        sortorder = dim.argsort()

        out = DataBlock()
        out.blockname = self.blockname
        out.comments = self.comments
        out.xname = self.xname

        for v in self.vardata.keys()[:]:
            data = self.vardata[v].view()
            try:
                i = data.dims_index(dimname)
                out[v] = data.take(sortorder, i)
            except IndexError:  # This variable does not contain sort direction
                out[v] = data
                continue

        olddim = self.ivardata[dimname]
        out.xname = self.xname
        data = self[dimname][sortorder]
        out.replace_dim(olddim, olddim.__class__(olddim, data=data))
        return out

    def var_report(self, name):
        #local import to avoid circular import
        from hftools.constants import SIFormat
        try:
            self.ivardata[name].data + 0
            isnumber = True
        except TypeError:
            isnumber = False
        tofmt = dict(name=name)
        tofmt["N"] = "<%s>" % (len(self.ivardata[name].data.flat))

        if isnumber:
            fmt = SIFormat(unit=self.ivardata[name].unit, digs=None)
            tofmt["max"] = fmt % self.ivardata[name].data.max()
            tofmt["min"] = fmt % self.ivardata[name].data.min()
            unit = self.ivardata[name].unit
            tofmt["unit"] = "[%s]" % (unit if unit is not None else "")
            fmt = "%(name)-15s %(unit)-5s %(N)-6s min: %(min)13s  "\
                  "max: %(max)13s "
        else:
            tofmt["dtype"] = "{%-5s}" % self.ivardata[name].data.dtype
            fmt = "%(name)-15s %(dtype)-5s %(N)-6s"
        return fmt % tofmt

    def depvar_report(self, name):
        v = self.vardata[name]
        shape = "<%s>" % ("x".join([str(x) for x in v.shape[:]]))
        unit = self.vardata[name].unit
        unit = "[%s]" % (unit if unit is not None else "")
        return dict(name=name, shape=shape, unit=unit)

    def report(self):
        report_strings = [u"Blockname: %s" % self.blockname]
        if self.ivardata.keys():
            sweepvars = u"\n".join([self.var_report(name)
                                   for name in self.ivardata.keys()])
            report_strings.append(u"sweep vars:\n%s" % sweepvars)
        if self.vardata:
            fmt = u"%(name)-15s %(unit)-5s %(shape)-12s "
            depvars = u"\n".join([fmt % self.depvar_report(name)
                                  for name in self.vardata])
            report_strings.append(u"Dependent vars:\n%s" % depvars)
        return (u"\n\n").join(report_strings)

    def __str__(self):
        return cast_str(self.report())

    def __repr__(self):
        return cast_str(self.report())

    def __dir__(self):  # pragma: no cover
        elemnames = []
        for k, v in self.vardata.iteritems():
            if ismatrix(v):
                if v.shape[-2] <= 10 and v.shape[-1] <= 10:
                    for i in range(v.shape[-2]):
                        for j in range(v.shape[-1]):
                            elemnames.append("%s%s%s" % (k, i + 1, j + 1))
        return (self.allvarnames + elemnames + dir(type(self)) +
                self.__dict__.keys())

    def guess_units(self, guessunit=True):
        if guessunit:
            if isinstance(guessunit, (str, unicode)):
                varnames = [guessunit]
            elif isinstance(guessunit, (list, tuple, set, dict)):
                varnames = list(guessunit)
            else:
                varnames = self.allvarnames
                if self.comments:
                    varnames.extend(self.comments.property)
            for varname in varnames:
                unit = guess_unit_from_varname(varname)
                if varname in self and self[varname].unit is None:
                    if varname in self.ivardata:
                        olddim = self.ivardata[varname]
                        newdim = olddim.__class__(olddim, unit=unit)
                        self.replace_dim(olddim, newdim)
                    else:
                        self[varname].unit = unit
                if self.comments:
                    prop = self.comments.property
                    if varname in prop and prop[varname]:
                        v = prop[varname]
                        if getattr(v, "unit", "") is None:
                            v.unit = unit

    def remove_rep(self, newdimname="AllReps"):
        out = DataBlock()
        out.comments = self.comments
        dims = [x for x in self.ivardata.values() if isinstance(x, DimRep)]
        fullshape = tuple(x.data.shape[0] for x in dims)
        newdim = DimRep(newdimname, np.arange(np.multiply.reduce(fullshape)))
        for idx, dim in enumerate(dims):
            data = zeros(fullshape, dtype=dims[idx].data.dtype)
            a = hfarray(data, dims=dims)
            sliceidx = ((None,) * idx + (slice(None),) +
                        (None,) * (len(fullshape) - 1 - idx))
            a[...] = dim.data[sliceidx]
            a = np.array(a, copy=False)
            a.shape = (newdim.data.shape[0],)
            out[dim.name] = hfarray(a, dims=(newdim,))
        for x in self.vardata:
            out[x] = remove_rep(self[x], newdimname)
        return out

    def hyper(self, dimnames, replacedim, indexed=False, all=True):
        if isinstance(replacedim, (str, unicode)):
            replacedim = self.ivardata[replacedim]
        out = DataBlock()
        out.blockname = self.blockname
        dims = []
        sortdata = []
        for x in dimnames:
            if indexed:
                newname = "%s_index" % x
            else:
                newname = x
            dims.append(DimSweep(newname, stable_uniq(self[x]),
                                 unit=self[x].unit,
                                 outputformat=self[x].outputformat))
            sortdata.append(self[x])
        dims = tuple(dims)
        dims_shape = tuple(len(x.data) for x in dims)

        sortdata.append(range(len(sortdata[0])))

        sortorder = sorted(zip(*sortdata))
        sortorderidx = [x[-1] for x in sortorder]

        for dim in dims:
            out.ivardata[dim.name] = dim

        for k, v in self.vardata.iteritems():
            if replacedim not in v.dims:
                if all:
                    out[k] = v
                1 + 0  # coverage.py bug necessary to get coverage for
                       # continue on next line
                continue
            if k in out.ivardata:
                continue
            i = v.dims_index(replacedim)
            v = hfarray(v.take(sortorderidx, axis=i), copy=False, order="C")
            new_shape = v.shape[:i] + dims_shape + v.shape[i + 1:]
            v.shape = new_shape
            v.dims = v.dims[:i] + dims + v.dims[i + 1:]
            out[k] = v

        for k, v in self.ivardata.items():
            if k not in out and k != replacedim.name:
                out[k] = v
        return out

    def squeeze(self):
        db = DataBlock()
        db.blockname = self.blockname
        db.comments = self.comments
        for v in self.vardata.keys():
            db[v] = self[v].squeeze()
        for k, v in self.ivardata.items():
            db[k] = v
        return db

    def interpolate(self, variable, defaultmode=None):
        out = DataBlock()
        for k, v in self.vardata.items():
            out[k] = interpolate(variable, self[k], defaultmode)
        return out


def interpolate(newx, y, defaultmode=None):
    mode = getattr(y, "interpolationmode", defaultmode)
    if isinstance(newx, (_hfarray,)):
        if len(newx.dims) != 1:
            raise ValueError("hfarray must have one dimension")
        else:
            newx = newx.dims[0]
    if newx not in y.dims:
        return y
    oldx = y.dims.get_matching_dim(newx)
    olddims = y.dims
    dimidx = olddims.matching_index(newx)
    newdims = olddims[:dimidx] + (newx,) + olddims[dimidx + 1:]
    if mode in [None, "none"]:
        bools = np.array(newx.data)[:, None] == np.array(oldx.data)[None, :]
        boolarray = hfarray(bools.any(axis=0),
                            dims=(oldx,),
                            unit=y.unit,
                            outputformat=y.outputformat)
        if boolarray.sum() != len(newx.data):
            raise ValueError("Missing x-values")
        data = np.array(y, copy=False)[boolarray]
    elif mode in ["linear"]:
        data = interp1d(y.dims[dimidx].data, y, axis=dimidx)(newx.data)
    else:
        raise ValueError("Interpolation mode %r unknown" % mode)
    return hfarray(data, dims=newdims, unit=y.unit,
                   outputformat=y.outputformat)


try:
    from scipy.interpolate import interp1d
except ImportError:  # pragma: no cover
    def interp1d(*k, **kw):
        raise ImportError("Need scipy to do linear interpolation")


if __name__ == '__main__':

    """
    Blockname: RFmeas.LDMOS69.Spar

    sweep vars:
    freq            <201>  min:    50000000.0  max: 20050000000.0
    vd              <6>    min:           0.0  max:           5.0
    vg              <6>    min:           0.0  max:           5.0
    i               <2>    min:             0  max:             1
    j               <2>    min:             0  max:             1

    Dependent vars:
    H               <6x6x201x2x2>
    Id              <6x6x201>
    Ig              <6x6x201>
    S               <6x6x201>
    mason           <6x6x201>
    vd              <6x6x201>
    vg              <6x6x201>
    """

    def complex_normal(loc=0, scale=1, dims=None):
        size = [x.data.shape[0] for x in dims]
        result = np.empty(size, np.complex128)
        result[:].real = rnd.normal(loc, scale, size)
        result[:].imag = rnd.normal(loc, scale, size)
        return hfarray(result, dims=dims)

    def normal(loc=0, scale=1, dims=None):
        size = [x.data.shape[0] for x in dims]
        return hfarray(rnd.normal(loc, scale, size), dims=dims)

    db = DataBlock()
    db.blockname = "RFmeas.LDMOS69.Spar"
    freq = DimSweep("freq", linspace(50e6, 20.05e9, 201))
    vd = DimSweep("vd", linspace(0, 5, 6))
    vg = DimSweep("vg", linspace(0, 5, 6))
    sweep_dims = (vd, vg, freq)
    matrix_dims = (DimMatrix_i("i", 2), DimMatrix_j("j", 2))

    db.freq = freq
    db.vd = vd
    db.vg = vg

    db.H = complex_normal(dims=sweep_dims + matrix_dims)
    db.Id = normal(dims=sweep_dims)
    db.Ig = normal(dims=sweep_dims)
    db.S = complex_normal(dims=sweep_dims)
    db.Ig = normal(dims=sweep_dims)
    db.Ig = normal(dims=sweep_dims)
    db.mason = complex_normal(dims=sweep_dims)
    db.vd = normal(dims=sweep_dims)
    db.vg = normal(dims=sweep_dims)

    print(db.report())
    from hftools.file_formats.spdata import read_spdata
    dc = read_spdata("../io/tests/testdata/sp_oneport_2_1.txt")
