#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os
import types
from .path import path

def add_method_to_Path(method):
    setattr(path, method.__name__.strip("_"), method)
    return method

#class path(path):
@add_method_to_Path
def _isdir(filename):
    return os.path.isdir(filename)
