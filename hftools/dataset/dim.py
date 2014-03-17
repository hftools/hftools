# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
u"""
dim
========
.. autoclass:: DimBase



"""
import datetime
import numpy as np

from hftools.utils import is_numlike, is_integer, deprecate
from hftools.py3compat import integer_types

def dims_has_complex(dims):
    for dim in reversed(dims):
        dims = (ComplexDerivAxis, ComplexIndepAxis, ComplexDiagAxis)
        if isinstance(dim, dims):
            return True
    return False


def info_has_complex(info):
    deprecate("info_has_complex is deprecated")
    return dims_has_complex(info)


def flatten(sequence):
    for item in sequence:
        if isinstance(item, (list, tuple)):
            for subitem in flatten(item):
                yield subitem
        else:
            yield item


class DimBase(object):
    sortprio = 0

    def __init__(self, Name, data=None, unit=None, name=None,
                 outputformat=None):
        if isinstance(Name, DimBase):
            dim_data = Name.data
            dim_name = Name.name
            dim_unit = Name.unit
            dim_outputformat = Name.outputformat
        else:
            dim_data = data
            dim_name = Name
            dim_unit = unit
            dim_outputformat = outputformat

        if data is not None:
            dim_data = data
        if unit is not None:
            dim_unit = unit
        if name is not None:
            dim_name = name
        if outputformat is not None:
            dim_outputformat = outputformat

        if isinstance(dim_data, integer_types):
            dim_data = list(range(dim_data))

        if hasattr(dim_data, "tolist"):
            dim_data = [dim_data.tolist()]
        if not isinstance(dim_data, (list, tuple)):
            dim_data = list(dim_data)

        self._data = tuple(flatten(dim_data))
        self._name = dim_name
        self._unit = dim_unit
        self._outputformat = dim_outputformat

    @property
    def data(self):
        if len(self._data) == 0:
            d = np.array([])
        elif isinstance(self._data[0], (np.datetime64, datetime.datetime)):
            d = np.asarray(self._data, np.dtype("datetime64[us]"))
        else:
            d = np.asarray(self._data)
        return d

    @property
    def name(self):
        return self._name

    @property
    def unit(self):
        return self._unit

    @property
    def outputformat(self):
        if self._outputformat is None:
            if is_integer(self._data[0]):
                return "%d"
            elif is_numlike(self._data[0]):
                return "%.16e"
            else:
                return "%s"
        return self._outputformat

    def fullsize(self):
        return self.data.shape[0]

    def __hfarray__(self):
        return (self.data, (self,))

    def __lt__(self, other):
        a = (self.sortprio, self.name, self.__class__, self._data,
             self._unit, self._outputformat)
        try:
            b = (other.sortprio, other.name, other.__class__, other._data,
                 other._unit, self._outputformat)
        except AttributeError:
            a = self.name
            b = other
        return a < b

    def __eq__(self, other):
        a = (self.sortprio, self.name, self.__class__, self._data,
             self._unit, self._outputformat)
        try:
            b = (other.sortprio, other.name, other.__class__, other._data,
                 other._unit, self._outputformat)
        except AttributeError:
            a = self.name
            b = other
        return a == b


    def __repr__(self):
        return "%s(%r, shape=%r)" % (self.__class__.__name__,
                                     self.name,
                                     self.data.shape)

    def __hash__(self):
        return hash((self.sortprio, self.name, self.__class__, self._data,
                     self._unit, self._outputformat))

    def __getitem__(self, index):
        if isinstance(index, slice) and (index == slice(None, None, None)):
            return self
        elif isinstance(index, slice):
            a = self.__class__(self.name, self.data[index], unit=self.unit,
                               outputformat=self.outputformat)
            return a
        elif isinstance(index, np.ndarray):
            a = self.__class__(self.name, self.data[index], unit=self.unit,
                               outputformat=self.outputformat)
            return a
        else:
            raise IndexError("Must index with slice")

    def copy(self):
        return self


class _DiagMeta(type):
    def __init__(cls, *args):
        if cls.__name__ != "_DiagAxis":
            if not hasattr(cls._indep_axis, "_diag_axis"):
                cls._indep_axis._diag_axis = cls
                cls._indep_axis._deriv_axis = cls._deriv_axis
            if not hasattr(cls._deriv_axis, "_diag_axis"):
                cls._deriv_axis._diag_axis = cls
                cls._deriv_axis._indep_axis = cls._indep_axis


class _DiagAxis(DimBase):
    __metaclass__ = _DiagMeta
    sortprio = 0

    @property
    def indep_axis(self):
        return self._indep_axis(self)

    @property
    def deriv_axis(self):
        return self._deriv_axis(self)


class _IndepAxis(DimBase):
    @property
    def diag_axis(self):
        return self._diag_axis(self)

    @property
    def deriv_axis(self):
        return self._deriv_axis(self)


class _DerivAxis(DimBase):
    @property
    def diag_axis(self):
        return self._diag_axis(self)

    @property
    def indep_axis(self):
        return self._indep_axis(self)


class IndepAxis(_IndepAxis):
    pass


class DerivAxis(_DerivAxis):
    pass


class DiagAxis(_DiagAxis):
    _indep_axis = IndepAxis
    _deriv_axis = DerivAxis


class DimAnonymous(DiagAxis):
    pass


class DimSweep(DiagAxis):
    pass


class DimRep(DiagAxis):
    sortprio = 1


class ComplexIndepAxis(_IndepAxis):
    sortprio = 2001
    pass


class ComplexDerivAxis(_DerivAxis):
    sortprio = 2002
    pass


class ComplexDiagAxis(_DiagAxis):
    sortprio = 2000
    _indep_axis = ComplexIndepAxis
    _deriv_axis = ComplexDerivAxis

CPLX = (ComplexIndepAxis("cplx", 2), ComplexDerivAxis("cplx", 2))


class _DimMatrix(DiagAxis):
    sortprio = 1000


class DimMatrix_Indep_i(_DimMatrix):
    sortprio = 1000


class DimMatrix_Deriv_i(_DimMatrix):
    sortprio = 1000


class DimMatrix_i(_DimMatrix):
    _indep_axis = DimMatrix_Indep_i
    _deriv_axis = DimMatrix_Deriv_i
    sortprio = 1000


class DimMatrix_Indep_j(_DimMatrix):
    sortprio = 1000


class DimMatrix_Deriv_j(_DimMatrix):
    sortprio = 1000


class DimMatrix_j(_DimMatrix):
    _indep_axis = DimMatrix_Indep_j
    _deriv_axis = DimMatrix_Deriv_j
    sortprio = 1001


class DimPartial(DimSweep):
    pass



