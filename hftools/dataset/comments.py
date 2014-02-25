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
import re

import numpy as np

from hftools.constants import convert_with_unit,\
    string_number_with_unit_to_value

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

reg_datetime = re.compile("[0-9]{4}-[0-1][0-9]-[0-3][0-9][ \t]+"
                          "[0-2][0-9]:[0-5][0-9](:[0-5][0-9]([.][0-9]*)?)?")


def conv_date_time(value):
    value = " ".join(value.strip().split())
    if reg_datetime.match(value):
        return np.datetime64(value)
    else:
        raise ValueError


def process_comment(comment):
    reg = reg_comment_all.match(comment)
    out = {}
    if reg:
        varname, _, unit, value = reg.groups()
        if unit is None:
            try:
#                value = string_number_with_unit_to_value(value)
                converters = [string_number_with_unit_to_value,
                              conv_date_time]
                value = convert(converters, value)
            except ValueError:
                pass
        else:
            value = convert([convert_with_unit], unit, value)
        out[varname] = value
    return out


class Comments(object):
    comment_reg = re.compile(r"([^:]+)[ \t]*[:][ \t]*(.*)")
    spacing_reg = re.compile(r"[ \t]+")
    width = 76

    def __init__(self, fullcomments=None, **kw):
        self.fullcomments = []
        self.property = {}
        self.property.update(kw)
        if fullcomments is not None:
            self.add_from_comments(fullcomments)

    def add_from_comment(self, comment):
        comment = comment.lstrip("!").strip()
        if comment:
            for key, v in process_comment(comment).iteritems():
                self.property[key] = v
            self.fullcomments.append(comment)

    def add_from_comments(self, comments):
        for comment in comments:
            self.add_from_comment(comment)

    def extend(self, comment):
        self.fullcomments.extend(comment.fullcomments)
        for k, v in comment.property.iteritems():
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
        table = [(" Comments ".center(self.width + 1, "-"),),
                 (keyfmt % ("Key".center(keycolwid, " ")),
                  valuefmt % ("Value".center(valuecolwid, " "))),
                 ("-" * keycolwid,
                  "-" * valuecolwid)]
        for key, value in sorted(self.property.iteritems()):
            if hasattr(value, "strip"):
                value = value.strip()[:valuecolwid]
            table.append((keyfmt % key, valuefmt % (value)))
            key = ""
        return [" ".join(x) for x in table]

    def __repr__(self):
        return "\n".join(self.table())
