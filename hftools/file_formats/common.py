# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

u"""
hftools.file_formats.common
================

"""
from __future__ import print_function
import re
import itertools
import time
import datetime
import numpy as np
from hftools.dataset import hfarray, ismatrix, DataBlock
from hftools.constants import convert_with_unit,\
    string_number_with_unit_to_value
from hftools.utils import to_numeric

_examples = """
!START Time : 2010-10-19 16:01:26
!END Time   : 2010-10-19 16:04:26

!Measurement Setups
!  Project         : Waveguide Calibration
!  DUT             : OML WR-03 Cal kit
!  Calibration     : No Cal - Raw Data
!  BiasCondition   : No Biasing/Passive Devices
!  CablePort1      : Cable at Port 1
!  CablePort2      : Cable at Port 2
!  AdapterPort1    : Adapter at Port 1
!  AdapterPort2    : Adapter at Port 2
!  ProbePort1      : OML Waveguide Extender WR03
!  ProbePort2      : OML Waveguide Extender WR03
!  ProbeStation    : N/A
!  PNA             : Agilent E8361A
!  ExtenderModule  : OML Extender
!  RepeatSweepWait : 10
!  RepeatSweepNum  : 1
!  DeviceID        : Unknown

!PNA settings read from the instrument
!  Sweep_Type               [ ]     : LIN
!  Sweep_Point              [#]     : 201
!  Average_Stat             [1=ON]  : 0
!  Average_Count            [#]     : 1
!  Start_Frequency          [Hz]    : +2.20000000000E+011
!  Stop_Frequency           [Hz]    : +3.25000000000E+011
!  IF_Bandwidth             [Hz]    : 100
!  RF_Power1                [dBm]   : -10
!  RF_Power2                [dBm]   : -10
!  RF_Slope_Stat            [1=ON]  : 0
!  RF_Slope                 [dB/GHz]    : 0
!  RF_Atten1                [dB]    : 0
!  RF_Atten2                [dB]    : 0
!  PortExtension_Stat       [1=ON]  : 0
!  Port1Ext_Time            [sec]   : 0
!  Port2Ext_Time            [sec]   : 0
!  Correction_Stat          [1=ON]  : 0

!Measurement data for connection # 1  sweep num 1
!Created: Fri Jun 11 09:53:19 2010
!Date: 2010-04-20
!Power[dBm]: 1.2
!Date: 2002-10-25, 09:09
!Vgs = 2.000000E-1
!Vds = 1.500000E+0
!Ig = 1.579100E-6
!Id = 3.120300E-2
!VNA mode="HP8510C raw waves"
!Power1=10.00 dBm
!Power2=10.00 dBm
!PortExtension1=0.000000 s
!PortExtension2=0.000000 s
!Averaging=32
!Attenuator1=0 dB
!Attenuator2=0 dB
!Slope1=0.0000 dB/GHz
!Slope2=0.0000 dB/GHz
!Testsetaddress=20
!Smoothing=5.00 %
!Date=3229918022.150000
!Datestring=2006-05-08 09:27:02
!Instrument list=
!Calibration: TRL
!Cable port 1:
!Cable port 2:
!Adapter port 1:
!Adapter port 2:
!Customer=
!Description of DUT=Type N male load  SP503388 S/N 00923
!Project=
!ClimateDateString=2006-05-08 09:26:11
!ClimateDate=3229917971
!MV_8_104_GT1  [C]=23.8
!MV_8_104_GT3  [C]=23.1
!MV_8_104_GF1  [%]=29
"""

reg_comment = re.compile(r"\s*(.*?)\s*([[]([%\w\d:=_]+)[]])?"
                         r"\s*[:=]\s*([^ \t]+)")
reg_comment_all = re.compile(r"\s*(.*?)\s*([[]([%\w\d:=_#\s/]+)[]])?"
                             r"\s*[:=]\s*(.+)")


def convert(convertors=[], *vars):
    for convfun in convertors:
        try:
            return convfun(*vars)
        except ValueError:
            pass
    raise ValueError

reg_time = re.compile("[0-2][0-9]:[0-5][0-9](:[0-5][0-9]([.][0-9]*)?)?")
reg_date = re.compile("[0-9]{4}-[0-1][0-9]-[0-3][0-9]")
reg_datetime = re.compile("[0-9]{4}-[0-1][0-9]-[0-3][0-9][ \t]+"
                          "[0-2][0-9]:[0-5][0-9](:[0-5][0-9]([.][0-9]*)?)?")


#reenable when arrays of no dims can set elements of datetime arrays
if hasattr(np, "datetime64"):
    def conv_date(value):
        value = " ".join(value.strip().split())
        if reg_date.match(value):
            return np.datetime64(value)
        else:
            raise ValueError

    def conv_date_time(value):
        value = " ".join(value.strip().split())
        if reg_datetime.match(value):
            return np.datetime64(value)
        else:
            raise ValueError
else:  # pragma: no coverage:
    def conv_date(value):
        timestamp = time.mktime(time.strptime(value, "%Y-%m-%d"))
        return datetime.date.fromtimestamp(timestamp)

    def conv_date_time(value):
        value = " ".join(value.strip().split())
        return datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def process_comment(comment):
    reg = reg_comment_all.match(comment)
    out = {}
    if reg:
        varname, _, unit, value = reg.groups()
        if unit is None:
            try:
#                value = string_number_with_unit_to_value(value)
                value = convert([string_number_with_unit_to_value,
                                 to_numeric,
                                 conv_date_time,
                                 conv_date], value)
            except ValueError:
                pass
        else:
            try:
                value = convert([convert_with_unit], unit, value)
            except ValueError:
                pass
        out[varname] = value
    return out


class Comments(object):
    comment_reg = re.compile(r"([^:]+)[ \t]*[:][ \t]*(.*)")
    spacing_reg = re.compile(r"[ \t]+")
    width = 76

    def __init__(self, fullcomments=None, **kw):
        if fullcomments is None:
            fullcomments = []
        self.fullcomments = fullcomments
        self.property = {}
        self.property.update(kw)
        self.add_from_comments(fullcomments)

    def add_from_comment(self, comment):
        comment = comment.lstrip("!").strip()
        if comment:
            for key, v in process_comment(comment).items():
                self.property[key] = v

    def add_from_comments(self, comments):
        for comment in comments:
            self.add_from_comment(comment)

    def extend(self, comment):
        self.fullcomments.extend(comment.fullcomments)
        for k, v in comment.property.items():
            self.property[k] = v

    def copy(self):
        out = Comments()
        out.fullcomments = self.fullcomments[:]
        out.property = self.property.copy()
        return out

    def table(self):
        if not self.property:
            return []
        keycolwid = max([len(x) for x in self.property.keys()])
        valuecolwid = self.width - keycolwid
        keyfmt = "%%-%ds" % keycolwid
        valuefmt = "%%-%ds" % (valuecolwid)
        table = [(" Comments ".center(self.width + 1, "-"), ),
                 (keyfmt % ("Key".center(keycolwid, " ")),
                  valuefmt % ("Value".center(valuecolwid, " "))),
                 ("-" * keycolwid,
                  "-" * valuecolwid)]
        for key, value in sorted(self.property.items()):
            if hasattr(value, "strip"):
                value = value.strip()
            table.append((keyfmt % key, valuefmt % (value)))
            key = ""
        return [" ".join(x) for x in table]

    def __repr__(self):
        return "\n".join(self.table())


_trtable = {'b1/a1 raw': 'b11',
            'b1/a2 raw': 'b12',
            'b2/a1 raw': 'b21',
            'b2/a2 raw': 'b22',
            'a1 raw': 'a11',
            'a1/a2 raw': 'a12',
            'a2/a1 raw': 'a21',
            'a2 raw': 'a22',
            'b1/a1': 'b11',
            'b1/a2': 'b12',
            'b2/a1': 'b21',
            'b2/a2': 'b22',
            'a1': 'a11',
            'a1/a2': 'a12',
            'a2/a1': 'a21',
            'a2': 'a22',
            'Freq[Hz]': 'freq',
            'Frequency': 'freq',
            'Freq': 'freq'}


remove_enclosing_function_reg = re.compile(r"^([a-zA-Z0-9_]+)\((.+)\)$")


def remove_enclosing_function(functionname, newpat="%s",
                              report_error=True):
    r"""Ta bort funktionsanrop i variabelnamn for namngiven funktion.

    Invariabler

        *functionname*
            Namn pa funktion vars anrop skall tas bort

        *newpat*
            strang monster som anvands for att bygga nytt variabelnamn

        *report_error*
            om exception skall rapporteras


    """
    regexp = remove_enclosing_function_reg

    def translator(old):
        res = regexp.match(old)
        if res and res.groups()[0] == functionname:
            new = newpat % res.groups()[1]
            return new
        else:
            return old
    return translator


def normalize_names(data):
    """Convert some standard column names to easier to use format.

    e.g. Freq[Hz] -> freq
         b1/a1 raw -> b11
    """
    out = data.copy()
    for old, new in _trtable.items():
        if old in data and new in data:
            continue
        out.rename(old, new)
    meantr = remove_enclosing_function("Mean")
    stdtr = remove_enclosing_function("Std", "s_%s")

    for trfunk in [meantr, stdtr]:
        for old in out.vardata:
            new = trfunk(old)
            if new == old:
                pass
            elif new in out.vardata:
                msg = ("%r already in data when trying to "
                       "normalize %r") % (new, old)
                raise ValueError(msg)
            else:
                out.rename(old, new)
    return out


def get_dims_names(*a):
    return tuple([x.name for x in a])


def make_col_from_matrix(header, columns, fmt_matrix, fortranorder=False):
    fullheader = []
    fullcolumns = []
    for colname, col in zip(header, columns):
        if ismatrix(col):
            for i in range(col.dims[-2].data.shape[0]):
                for j in range(col.dims[-1].data.shape[0]):
                    if fortranorder:
                        I, J = j, i
                    else:
                        I, J = i, j
                    fullheader.append(fmt_matrix % (colname, I + 1, J + 1))
                    fullcolumns.append(col[..., I, J])
        else:
            fullheader.append(colname)
            fullcolumns.append(col)
    return fullheader, fullcolumns


def format_elem(fmt, elem):
    if np.iscomplexobj(elem):
        return [fmt % elem.real, fmt % elem.imag]
    else:
        return [fmt % elem]

findpad_reg = re.compile("[%][-]?([0-9]+)")
def padhead(head, width):
    return head + " " * (max(len(head), width) - len(head))

def format_complex_header(header, columns, floatfmt,
                          realfmt, imagfmt, unit_fmt=True,
                          padheader=False):
    outheader = []
    for colname, col in zip(header, columns):
        res = findpad_reg.match(col.outputformat)
        if padheader and res:
            width = int(res.groups()[0])
        else:
            width = 0
        if np.iscomplexobj(col):
            value = format_unit_header(realfmt % colname, col, unit_fmt)

            outheader.append(padhead(value, width))
            if imagfmt is not None:
                value = format_unit_header(imagfmt % colname, col, unit_fmt)
                outheader.append(padhead(value, width))
        else:
            value = format_unit_header(floatfmt % colname, col, unit_fmt)
            outheader.append(padhead(value, width))
    return outheader


def format_unit_header(colname, col, use_unit_fmt=True):
    unit_fmt = "%s [%s]"
    nounitfmt = "%s"
    if col.unit and use_unit_fmt:
        return unit_fmt % (colname, col.unit)
    else:
        return nounitfmt % colname


def make_block_iterator(data, first=False):
    if not data.vardata:   # only x-values
        sweepvars = ()
        head = data.ivardata.keys()[0]
        dim = data.ivardata[head]

        header = [head, ]
        fmts = (data[head].outputformat,)
        columns = (hfarray(dim),)
        blockname = data.blockname
        comments = data.comments
        yield (sweepvars, header, fmts, columns,
               blockname, comments)
        raise StopIteration
    d = data.vardata.values()[0]
    if ismatrix(d):
        dimsnames = [x.name for x in d.dims[:-2]]
    else:
        dimsnames = [x.name for x in d.dims]
    dimsdata = [hfarray(data.ivardata[iname]) for iname in dimsnames]
    dimsidx = [range(x.shape[0]) for x in dimsdata]
    if first:
        if data.comments:
            comments = data.comments.copy()
            comments.property.clear()
        else:
            comments = Comments()
    else:
        comments = Comments()
    for enum, idx in enumerate(itertools.product(*dimsidx[1:])):
        sweepvars = []
        for iname, dim, i in zip(dimsnames[1:], dimsdata[1:], idx):
            fmt = getattr(dim, "outputformat", "%.16e")
            sweepvars.append((iname, fmt, dim[i]))
        if dimsnames:
            header = [dimsnames[0]] + data.vardata.keys()
            columns = dimsdata[:1] + [col[(slice(None), ) + idx]
                                      for col in data.vardata.values()]
            fmts = [getattr(x, "outputformat", "%.16e") for x in columns]
        else:
            header = []
            columns = []
            fmts = []
            for k, v in data.vardata.items():
                if v.dims == tuple():
                    comments.property[k] = v

        if enum:
            comments = Comments()
        yield (sweepvars, header, fmts, columns,
               data.blockname, comments)


def db_iterator(indb, fmt_block_generator):
    datablock = {}
    used_ivars = set()
    for vname in indb.vardata.keys():
        if ismatrix(indb.vardata[vname]):
            dimsnames = get_dims_names(*indb[vname].dims)[:-2]
            names = get_dims_names(*indb[vname].dims)[-2:]
            used_ivars = used_ivars.union(names)
        else:
            dimsnames = get_dims_names(*indb[vname].dims)

        q = datablock.setdefault(dimsnames, DataBlock())
        if q.comments is None:
            q.comments = indb.comments
        q[vname] = indb[vname]
        q[vname].outputformat = indb[vname].outputformat

    #
    #Collect dangling ivars
    #
    for names in datablock.keys():
        used_ivars = used_ivars.union(names)

    for vname in set(indb.ivardata).difference(used_ivars):
        q = datablock.setdefault((vname,), DataBlock())
        if q.comments is None:
            q.comments = Comments()
        q[vname] = indb.ivardata[vname]

    first = True

    #Handle block with scalar data first
    dbs = []
    if tuple() in datablock:
        dbs.append(datablock.pop(tuple()))
    for v in datablock.values():
        dbs.append(v)

    for db in dbs:
        for block_content in make_block_iterator(db, first):
            if first:
                first = False
            else:
                yield []
            for rad in fmt_block_generator(*block_content):
                yield rad


