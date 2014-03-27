# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
from __future__ import print_function, unicode_literals
import os
import six
import sys
import numpy as np
DEFAULT_ENCODING = "cp1252"


integer_types = six.integer_types + (np.integer,)
text_type = six.text_type
string_types = six.string_types


def no_code(x, encoding=None):
    return x


def decode(s, encoding=None):
    encoding = encoding or DEFAULT_ENCODING
    return s.decode(encoding, "replace")


def encode(u, encoding=None):
    encoding = encoding or DEFAULT_ENCODING
    return u.encode(encoding, "replace")


def cast_unicode(s, encoding=None):
    if isinstance(s, bytes):
        return decode(s, encoding)
    return s


def cast_bytes(s, encoding=None):
    if not isinstance(s, bytes):
        return encode(s, encoding)
    return s


if sys.version_info[0] >= 3:
    PY3 = True
    from functools import reduce
    from configparser import ConfigParser as SafeConfigParser
    from configparser import NoOptionError

    def reraise(tp, value, tb=None):
        raise tp(value).with_traceback(tb)
    import copyreg
    import builtins
    import itertools
    from subprocess import Popen, PIPE

    def popen(command):
        p = Popen(command, shell=True,
                  stdin=PIPE, stdout=PIPE, stderr=PIPE)
        return (p.stdin, p.stdout, p.stderr)
    filter = builtins.filter
    filterfalse = itertools.filterfalse
    from io import StringIO
    input = builtins.input
    print = builtins.print
    getcwdu = os.getcwd
else:
    PY3 = False
    from ConfigParser import SafeConfigParser
    from ConfigParser import NoOptionError
    six.exec_("""def reraise(tp, value, tb=None):
    raise tp, value, tb
""")
    import copy_reg as copyreg
    import os

    def popen(command):
        return os.popen3(command)
    import __builtin__
    reduce = __builtin__.reduce
    import itertools
    filter = itertools.ifilter
    filterfalse = itertools.ifilterfalse
    from cStringIO import StringIO
    input = raw_input
    getcwdu = os.getcwdu

    def print(*k, **kw):
        kw = kw.copy()
        if "flush" in kw:
            del kw["flush"]
        return __builtin__.print(*k, **kw)


def cast_str(s, encoding=None):
    if PY3:
        return cast_unicode(s, encoding)
    else:
        return cast_bytes(s, encoding)


