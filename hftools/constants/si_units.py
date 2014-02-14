# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import abc
import imp
import re
import sys

import numpy as np

import hftools.dataset
from hftools.utils import to_numeric, isrealnumber

__all__ = ["siprefixes", "unit_to_multiplier",
           "string_number_with_unit_to_value",
           "convert_with_unit", "SIFormat"]

siprefixes = dict(d=0.1, c=0.01,
                  m=1e-03, u=1e-06, n=1e-09,
                  p=1e-12, f=1e-15, a=1e-18,
                  k=1e003, M=1e006, G=1e009,
                  T=1e012, P=1e015, E=1e018)
siprefixes[""] = 1.
si_exp_to_prefixes = {0: "", 3: "k", 6: "M", 9: "G", 12: "T", 15: "P", 18: "E", 21: "Z", -24: "Y",
                      -3: "m", -6: "u", -9: "n", -12: "p", -15: "f", -18: "a", -21: "z", -24: "y"}

siunit_names = ["Hz", "s", "S", "Ohm", "m", "V", "A", "F", "W"]

siunits = imp.new_module("siunits")

for u in siunit_names:
    for p, val in siprefixes.iteritems():
        setattr(siunits, p + u, val)


def unit_to_multiplier(unit):
    """Convert a unit to multiplier with baseunit
        >>> unit_to_multiplier("ms")
        (0.001, 's')
        >>> unit_to_multiplier("GHz")
        (1000000000.0, 'Hz')
    """
    unit = unit.strip()
    if len(unit) == 1:
        if unit == "%":
            return .01, None
        if unit == "#":
            return 1, None
        else:
            return 1, unit
    elif len(unit) >= 2 and unit[0] in siprefixes:
        if unit[1:] in siunit_names:
            return siprefixes[unit[0]], unit[1:]
        else:
            return 1, unit
    else:
        return 1, unit

reg_value = re.compile(r"[-+0-9.]+[eE]?[-+0-9]?")
reg_unit = re.compile(r"\s*[[]?([a-zA-Z%]+)[]]?")


def string_number_with_unit_to_value(string_number):
    if isrealnumber(string_number):
        return string_number
    res = reg_value.match(string_number)
    if res:
        value = float(res.group())
        reg_unit = re.compile(r"\s*[[]?([a-zA-Z%]+)[]]?")
        unit_res = reg_unit.match(string_number[res.span()[-1]:])
        if unit_res:
            unit = unit_res.group()
        else:
            unit = None
    else:
        msg = "Could not convert %r to value with unit" % (string_number)
        raise ValueError(msg)

    return convert_with_unit(unit, value)


def convert_with_unit(unit, value):
    if unit is None:
        return hftools.dataset.hfarray(value, unit=None)
    unit = unit.strip()
    if unit:
        mul, unit = unit_to_multiplier(unit)
    else:
        mul = 1
        unit = None
    return hftools.dataset.hfarray(to_numeric(value, error=True) * mul, unit=unit)


def _help_format_sci(value, digs):
    """
    >>> "%.4f, %d"%_help_format_sci(1.2345,2)
    '1.2345, 0'
    >>> "%.4f, %d"%_help_format_sci(1.2345e5,2)
    '123.4500, 3'
    >>> "%.4f, %d"%_help_format_sci(-1.2345e5,2)
    '-123.4500, 3'
    """
    m, e = mantissa(value)
    mul = e % 3
    if (e >= 0 and e < 3):
        return m * 10 ** mul, 0
    return (value * 10 ** -(e - mul), e - mul)


def mantissa(x):
    u"""Berakna mantissa och exponent av *x*

        >>> mantissa(125)
        (1.25, 2.0)
        >>> mantissa(0.0125)
        (1.25, -2.0)
        >>> mantissa(0.0)
        (0.0, 0.0)
        >>> mantissa(-125)
        (-1.25, 2.0)
        >>> mantissa(-0.0125)
        (-1.25, -2.0)
        >>> mantissa(0.0)
        (0.0, 0.0)

    """
    if x == 0:
        return 0., 0.

    exponent = np.floor(np.log10(np.sign(x) * x))
    mantissavalue = x / (10 ** exponent)
    return mantissavalue, exponent


def format_number(number, fmt="%(num)s %(unit)s", unit="", digs=3):
    if np.isinf(number):
        prefix = ""
        if number > 0:
            num = "inf"
        else:
            num = "-inf"
    else:
        num, exp = _help_format_sci(number, digs)
        if digs is None:
            pass
        else:
            numfmt = "%%.%df" % digs
            num = numfmt % num
        prefix = si_exp_to_prefixes[exp]
    prefixedunit = "%s%s" % (prefix, unit)
    return fmt % dict(num=num, unit=prefixedunit)


class SIFormat(object):
    def __init__(self, fmt="%(num)s %(unit)s", unit="", digs=3):
        self.fmt = fmt
        if unit is None:
            unit = ""
        self.unit = unit
        self.digs = digs

    def __mod__(self, value):
        return format_number(value, self.fmt, self.unit, self.digs)
