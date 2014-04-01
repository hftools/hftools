# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import abc
import datetime
import glob as glob_module
import os
import re
import warnings


import numpy as np

from hftools.core.exceptions import HFToolsWarning, HFToolsDeprecationWarning
from hftools.py3compat import PY3


def reset_hftools_warnings():
    try:
        __warningregistry__.clear()
    except NameError:
        pass


def timestamp(highres=False):
    """Return time stamp string for current time

    Example: "20121026T080312"
    """
    if highres:
        return datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f")[:-3]
    else:
        return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def warn(msg):
    warnings.warn(msg, HFToolsWarning)


def deprecate(msg):
    warnings.warn(msg, HFToolsDeprecationWarning)

if PY3:
    def isnumber(x):
        return isinstance(x, (int,
                              float,
                              complex))

    def isrealnumber(x):
        return isinstance(x, (int, float))
else:
    def isnumber(x):
        return isinstance(x, (abc.types.IntType,
                              abc.types.FloatType,
                              abc.types.ComplexType))

    def isrealnumber(x):
        return isinstance(x, (abc.types.IntType, abc.types.FloatType))


def is_numlike(a):
    if isinstance(a, np.ndarray):
        return np.issubdtype(a.dtype, np.number)
    else:
        return np.issubdtype(np.array(a).dtype, np.number)


def is_integer(a):
    if isinstance(a, np.ndarray):
        return a.dtype.kind == "i"
    else:
        return isinstance(a, int)


dateregexp = re.compile("^[0-9]{4}-[01][0-9]-[0123][0-9]"
                        "[ \t]*[0-9]{2}:[0-9]{2}(:[0-9]{2})?$")


def to_numeric(value, error=False):
    """Translate string *value* to numeric type if possible
    """
    try:
        return value + 0
    except TypeError:
        pass
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            try:
                if dateregexp.match(value):
                    return np.datetime64(value)
                elif value == "":
                    raise ValueError
            except (ValueError, TypeError):
                if error:
                    raise
                else:
                    return value
    return value


def stable_uniq(data):
    """return list of unique elements in the order they first appear in data
    """
    out = []
    for x in data:
        if x not in out:
            out.append(x)
    return out


def uniq(data):
    """return sorted list of unique elements
    """
    if hasattr(data, "flat"):
        return sorted(set(data.flat))
    else:
        return sorted(set(data))


def chop(x, rtol=1e-9, atol=1e-15):
    """Set small numbers to zero.

       *rtol* Tolerance relative maximum value of array
       *atol* Absolute tolerance
    """
    x = np.asanyarray(x)
    out = np.array(x)
    x = abs(x)
    out[(x < atol) | (x < (rtol * x.max()))] = 0
    return out


def make_dirs(*pathsegments):
    """Skapa directory fran *pathsegments*, skulle directoryn finnas
    hander inget::

       make_dirs('c:/kalle')      -> skapar c:\kalle\
       make_dirs('c:/', 'kalle')  -> skapar c:\kalle\
    """
    x = os.path.join(*pathsegments)
    if not os.path.isdir(x):
        os.makedirs(x)

reg_split = re.compile("[^0-9]+|[0-9]+")


def split_num(string):
    return list(map(to_numeric, reg_split.findall(string))), string


def lex_order(lista):
    ordered = sorted(list(map(split_num, lista)))
    return [orig for _, orig in ordered]


def glob(pat):
    return lex_order(glob_module.glob(pat))


try:  # pragma: no cover
    import winsound

    def beep(freq=None, duration=50):
        """Play sound.

        If *freq* is None then use SystemQuestion sound otherwise use
        parameters.

            *freq*  37-32767 Hz
            *duration* number of milliseconds to play sound.
        """
        if freq is None:
            winsound.PlaySound("SystemQuestion", winsound.SND_ALIAS)
        else:
            winsound.Beep(freq, duration)
except ImportError:  # pragma: no cover

    def beep(freq=None, duration=50):
        """Place holder empty function on this platform
        """
        pass


def flatten(lista):
    """Plattar ut list av listor så den endast har en nivå.

        >>> flatten([1,[2,3,[4]]])
        [1, 2, 3, 4]
        >>> flatten([1,2,3,4])
        [1, 2, 3, 4]
        >>> flatten([])
        []
        >>> flatten([[[]]])
        []
    """
    nylista = []
    for x in lista:
        if type(x) in (list, tuple):
            nylista += flatten(x)
        else:
            nylista += [x]
    return nylista


