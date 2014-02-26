# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

class HFToolsIOError(IOError):
    pass


class DimensionMismatchError(ValueError):
    pass


class HFArrayShapeDimsMismatchError(ValueError):
    pass


class HFArrayError(ValueError):
    pass
