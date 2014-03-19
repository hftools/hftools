# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

u"""
CITI File Format
==================

    .. autofunction:: read_citi
    .. autofunction:: save_citi
    .. autoexception:: CITIFileError
"""
from __future__ import print_function
import re

import numpy as np
from numpy import array, iscomplexobj, pi
import hftools.dataset
from hftools.dataset import DataBlock, DimSweep, hfarray, _DimMatrix
from hftools.core.exceptions import HFToolsIOError
from hftools.file_formats.common import Comments

from hftools.file_formats.readbase import ManyOptional, One, Token,\
    ReadFileFormat, Optional, OneOf
from hftools.utils import is_numlike

reg_outerfun = re.compile(r"^([A-Za-z_][A-Za-z_0-9]*)\((.*)\)(.*)")
reg_num_unit = re.compile(r"^[ \t]*([-0-9.eE]+) ?"
                          r"([dcmunpfakMGTPE]?[A-Za-z/]+)$")


class CITIFileError(HFToolsIOError):
    pass

reg_data = re.compile("[-0-9eE.]+")


class ReadCITIFileFormat(ReadFileFormat):
    match_matrix_element_name = [re.compile("([A-Za-z_]+)[[]([0-9]),"
                                            "([0-9])[]]$").match,
                                 re.compile("([A-Za-z_]+)([0-9])"
                                            "([0-9])$").match]

    def tokenize(self, stream):
        """Split stream of lines in sp-data format into
        stream of tagged lines.

        input:
        *stream*    stream of lines in sp-data format

        output:
                    stream of tagged lines
                    (tag, line) where tag is one of
                    "Comment", "Header", and "Data"
        """
        for idx, rad in enumerate(stream):
            lineno = idx + 1
            if self.verbose:
                print("\r               \r", lineno, end="")
            rad = rad.lstrip()
            if not rad:  # skip empty lines
                continue
            elif rad.startswith("NAME"):
                yield Token("NAME", lineno, rad[4:].strip())
            elif rad.startswith("VAR "):
                yield Token("VARDEF", lineno, rad[3:].strip().split())
            elif rad.startswith("DATA"):
                yield Token("DATADEF", lineno, rad[4:].strip().split())
            elif rad.startswith("VAR_LIST_BEGIN"):
                yield Token("VAR_LIST_BEGIN", lineno, rad)
            elif rad.startswith("VAR_LIST_END"):
                yield Token("VAR_LIST_END", lineno, rad)
            elif rad.startswith("SEG "):
                yield Token("SEG", lineno, rad)
            elif rad.startswith("SEG_LIST_BEGIN"):
                yield Token("SEG_LIST_BEGIN", lineno, rad)
            elif rad.startswith("SEG_LIST_END"):
                yield Token("SEG_LIST_END", lineno, rad)
            elif rad.startswith("BEGIN"):
                yield Token("DATA_LIST_BEGIN", lineno, rad)
            elif rad.startswith("END"):
                yield Token("DATA_LIST_END", lineno, rad)
            elif reg_data.match(rad):
                yield Token("DATA", lineno, rad)
            elif rad.startswith("!"):
                yield Token("COMMENT", lineno, rad[1:].strip())
            elif rad.startswith("#"):
                yield Token("COMMENT", lineno, rad[1:].strip())
            elif rad.startswith("COMMENT"):
                yield Token("COMMENT", lineno, rad[8:].strip())
            elif rad.startswith("CONSTANT"):
                yield Token("COMMENT", lineno, rad[:].strip())
            elif rad.startswith("CITIFILE"):
                yield Token("CITIFILE", lineno, rad[8:].strip())
            else:
                raise CITIFileError("Unknown linetype %r" % rad)
        else:  # pragma: no cover
            pass

    def parse_blocks(self, stream):
    # raise Exception("HANTERAR INTE KOMMENTARER INNE I MANYOPTIONAL")
        comments = []
        comments.extend(ManyOptional("COMMENT")(stream))
        citifileversion = Optional("CITIFILE")(stream)
        comments.extend(ManyOptional("COMMENT")(stream))
        name = One("NAME", "Must have END after DATA")(stream)
        comments.extend(ManyOptional("COMMENT")(stream))
        varnames = ManyOptional("VARDEF")(stream)
        comments.extend(ManyOptional("COMMENT")(stream))
        datanames = ManyOptional("DATADEF")(stream)
        comments.extend(ManyOptional("COMMENT")(stream))
        block = DataBlock()
        block.blockname = name[0]
        for name, typ, N in varnames:
            N = int(N)
            tagname, = OneOf(["VAR_LIST_BEGIN", "SEG_LIST_BEGIN"], "Missing VAR_LIST_BEGIN or SEG_LIST_BEGIN")(stream)
            if tagname.startswith("VAR_LIST"):
                datalist = handle_data(ManyOptional("DATA")(stream), typ)
                block[name] = DimSweep(name, datalist)
                One("VAR_LIST_END", "Missing VAR_LIST_END")(stream)
            elif tagname.startswith("SEG"):
                datalist, = One("SEG", "Missing SEG")(stream)
                _, start, stop, step = datalist.strip().split()
                block[name] = DimSweep(name,
                                       np.linspace(float(start),
                                                   float(stop),
                                                   int(step)))
                One("SEG_LIST_END", "Missing SEG_LIST_END")(stream)

        comments.extend(ManyOptional("COMMENT")(stream))
        for idx, com in enumerate(comments):
            if com.startswith("CONSTANT TIME"):
                date = tuple(com.strip().split()[2:])
                comments[idx] = "Measurement time: %s-%s-%s %s:%s:%s" % date
        block.comments = Comments(comments)
        shape = tuple(block.ivardata[i[0]].data.shape[0] for i in varnames)
        dims = tuple(block.ivardata[i[0]] for i in varnames)
        for name, typ in datanames:
            One("DATA_LIST_BEGIN", "Missing BEGIN")(stream)
            datalist = array(handle_data(ManyOptional("DATA")(stream), typ))
            datalist.shape = shape
            block[name] = hfarray(datalist, dims=dims)
            One("DATA_LIST_END", "Missing END")(stream)
        yield block

    def make_block_complex(self, block):
        return block


def handle_data(data, typ):
    if typ == "MAG":
        return [float(elem) for elem in data]
    elif typ == "RI":
        out = []
        for rad in data:
            r, i = rad.strip().split(",")
            out.append(float(r) + float(i) * 1j)
        return out
    elif typ == "MAGANGLE":
        out = []
        for rad in data:
            r, i = rad.strip().split(",")
            out.append(float(r) * np.exp(float(i) / 180 * pi * 1j))
        return out
    else:
        raise CITIFileError("Unknown Vartype %r" % (typ,))


def format_citi_block(inblock):
    block = DataBlock()
    for k, v in inblock.vardata.items():
        block[k] = v
    block.blockname = inblock.blockname
    block.comments = inblock.comments
    yield "CITIFILE A.01.01"

    if block.comments:
        for comment in block.comments.fullcomments:
            yield "!" + comment.lstrip("!")
    else:
        pass
    yield "NAME %s" % block.blockname

    for name, value in block.ivardata.items():
        if is_numlike(value.data) and not isinstance(value, _DimMatrix):
            yield "VAR %s MAG %s" % (name, value.data.shape[0])

    for name, value in block.vardata.items():
        if is_numlike(value):
            if hftools.dataset.ismatrix(value):
                names = []
                for i in hfarray(value.dims[-2]):
                    for j in hfarray(value.dims[-1]):
                        names.append("%s%s%s" % (name, i + 1, j + 1))
            else:
                names = [name]
            for n in names:
                if iscomplexobj(value):
                    yield "DATA %s RI" % (n)
                else:
                    yield "DATA %s MAG" % (n)

    for name, value in block.ivardata.items():
        if is_numlike(value.data) and not isinstance(value, _DimMatrix):
            yield "VAR_LIST_BEGIN"
            fmt = value.outputformat
            for rad in value.data:
                yield fmt % rad
            yield "VAR_LIST_END"

    for name, value in block.vardata.items():
        if is_numlike(value):
            if hftools.dataset.ismatrix(value):
                values = []
                for i in hfarray(value.dims[-2]):
                    for j in hfarray(value.dims[-1]):
                        values.append(value[..., i, j])
            else:
                values = [value]
            for val in values:
                yield "BEGIN"
                if iscomplexobj(val):
                    fmt = "%s,%s" % (val.outputformat, val.outputformat)
                    for rad in val.flat:
                        yield fmt % (rad.real, rad.imag)
                else:
                    fmt = val.outputformat
                    for rad in val.flat:
                        yield fmt % rad
                yield "END"


def is_citi(filename, rad):
    return rad.startswith("CITIFILE")


def save_citi(db, filename):
    """Write a Datablock to a sp-format file with name filename.
    """
    with open(filename, "w") as fil:
        for rad in format_citi_block(db):
            fil.write(rad)
            fil.write("\n")


def read_citi(filnamn, make_complex=True, property_to_vars=True,
              guess_unit=True, normalize=True, make_matrix=True,
              merge=True, verbose=False):
    return ReadCITIFileFormat.read_file(filnamn, make_complex=make_complex,
                                        property_to_vars=property_to_vars,
                                        guess_unit=guess_unit,
                                        normalize=normalize,
                                        make_matrix=make_matrix,
                                        merge=merge,
                                        verbose=verbose)

if __name__ == "__main__":

    a = read_citi("tests/testdata/citi/flush_thru.cti",
                  make_complex=True, property_to_vars=True,
                  guess_unit=True, normalize=True, make_matrix=True,
                  merge=True, verbose=False)

