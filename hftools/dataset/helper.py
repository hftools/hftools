# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

_varname_unit_guess_db = dict(Vds="V", Vgs="V", Vdg="V", Vgd="V",
                              vds="V", vgs="V", vdg="V", vgd="V",
                              VDS="V", VGS="V", VDG="V", VGD="V",
                              VDS_SET="V", VGS_SET="V", VDG_SET="V",
                              VGD_SET="V",
                              Vbe="V", Vce="V",
                              vbe="V", vce="V",
                              VBE="V", VCE="V",
                              VBE_SET="V", VCE_SET="V",
                              Id="A", Ig="A", Is="A",
                              id="A", ig="A",
                              ID="A", IG="A", IS="A",
                              Ib="A", Ie="A", Ic="A",
                              ib="A", ie="A", ic="A",
                              IB="A", IE="A", IC="A",
                              freq="Hz", Freq="Hz", FREQ="Hz",
                              f="Hz", F="Hz",
                              frequency="Hz", Frequency="Hz", FREQUENCY="Hz",
                              )

_varname_unit_guess_db['is'] = 'A'  # is is a keyword


def add_var_guess(varname, unit):
    global _varname_unit_guess_db
    _varname_unit_guess_db[varname] = unit


def guess_unit_from_varname(varname):
    return _varname_unit_guess_db.get(varname, None)


