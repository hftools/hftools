# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

u"""
SP-Data
=======

    .. autofunction:: read_spdata
    .. autofunction:: save_spdata
    .. autofunction:: normalize_names

"""
import io
import re
import numpy as np

from hftools.dataset import DataDict, DataBlock,  DimSweep, DimRep,\
    DimPartial, hfarray
from hftools.core.exceptions import HFToolsIOError
from hftools.file_formats.common import Comments, db_iterator,\
    make_col_from_matrix, format_complex_header, format_elem
from hftools.file_formats.readbase import ReadFileFormat
from hftools.file_formats.readbase import Token
from hftools.utils import to_numeric
from hftools.py3compat import string_types

class SPDataIOError(HFToolsIOError):
    pass

reg_num_unit = re.compile(r"^[ \t]*([-0-9.eE]+) " +
                          r"?([dcmunpfakMGTPE]?[A-Za-z/]+)$")
"""
def match_numunit(x):
    res = reg_num_unit.match(x)
    if res:
        value, unit = res.groups()
        multiplier, baseunit = unit_to_multiplier(unit)
        return float(value)*multiplier, baseunit
    else:
        return x
"""


reg_header = re.compile("^[A-Za-z_]")


class ReadSPFileFormat(ReadFileFormat):
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

        for radidx, rad in enumerate(stream):
            rad = rad.strip()
            lineno = radidx + 1
            if not rad:  # skip empty lines
                continue
            elif rad.startswith("!Fullcomments"):
                continue
            elif rad.startswith("!"):  # Comment line with information
                yield Token("Comment", lineno, rad[1:])
            elif reg_header.match(rad[0]):
                yield Token("Header", lineno, rad)
            else:
                yield Token("Data", lineno, rad)

    def group_blocks(self, stream):
        """Bunch tagged stream into sub blocks. Each sub block
        containing a measurement data block.
        """
        token, lineno, rad = next(stream)
        running = True
        while running:
            comments = []
            header = []
            data = []
            out = [comments, header, data]
            while token == "Comment":
                comments.append(rad)
                try:
                    token, lineno, rad = next(stream)
                except StopIteration:
                    raise SPDataIOError("File can not end in comments")
            if token == "Header":
                header.append(rad)
                try:
                    token, lineno, rad = next(stream)
                except StopIteration:
                    raise SPDataIOError("File can not end in header")
            else:
                raise SPDataIOError("Missing header")
            while running and (token == "Data"):
                data.append([to_numeric(x, False) for x in rad.split("\t")])
                try:
                    token, lineno, rad = next(stream)
                except StopIteration:
                    running = False
            yield out

    def parse_blocks(self, stream):
        for comments, header, data in stream:
            db = DataBlock()
            db.comments = Comments(comments)
            header = header[0].strip().split("\t")
            Nhead = len(header)
            #data = np.array(data)
            if Nhead != len(data[0]):
                msg = "Different number of header variables "\
                      "from data columns"
                raise SPDataIOError(msg)
            output = DataDict()
            for varname, column in zip(header, zip(*data)):
                output.setdefault(varname.strip(), []).append(column)
            for varname in output:
                data = output[varname]
                if len(data) > 1:
                    output[varname] = np.array(output[varname],
                                               order="F").T
                else:
                    output[varname] = np.array(output[varname][0])

            freq = DimSweep(header[0].strip(), output[header[0].strip()])
            db[header[0].strip()] = freq
            for x in output.keys()[1:]:
                if output[x].ndim == 1:
                    db[x] = hfarray(output[x], dims=(freq,))
                else:
                    repdim = DimRep("rep", output[x].shape[1])
                    db[x] = hfarray(output[x],
                                    dims=(freq, repdim)).squeeze()

            remove = []
            for vname in db.comments.property:
                if vname[:1] == "@":
                    unit = db.comments.property[vname].unit
                    data = [float(db.comments.property[vname])]
                    db[vname[1:]] = DimPartial(vname[1:], data=data, unit=unit)
                    remove.append(vname)
            for v in remove:
                del db.comments.property[v]
            db.comments.fullcomments = [com for com in db.comments.fullcomments
                                        if not com.startswith("@")]
            yield db


def format_sp_block(sweepvars, header, fmts, columns, blockname, comments):
    for comment in comments.fullcomments:
        yield ["!" + comment.lstrip("!")]
    for iname, fmt, value in sweepvars:
        yield [("!@%s=" + fmt) % (iname, value)]
    header, columns = make_col_from_matrix(header, columns, "%s%s%s")
    outheader = format_complex_header(header, columns,
                                      "%s", "Re(%s)", "Im(%s)",
                                      padheader=True)

    yield outheader
    fmts = [x.outputformat for x in columns]
    for row in zip(*columns):
        out = []
        for elem, fmt in zip(row, fmts):
            out.extend(format_elem(fmt, elem))
        yield out


def save_spdata(db, filename, encoding="cp1252"):
    """Write a Datablock to a sp-format file with name filename.
    """
    if isinstance(filename, string_types):
        fil = io.open(filename, "w", encoding=encoding)
    else:
        fil = filename
    for rad in db_iterator(db, format_sp_block):
        fil.write(u"\t".join(rad))
        fil.write(u"\n")

    if isinstance(filename, string_types):
        fil.close()


def read_spdata(filnamn, make_complex=True, property_to_vars=True,
                guess_unit=True, normalize=True, make_matrix=True,
                merge=True, hyper=False, verbose=False, encoding="cp1252"):
    return ReadSPFileFormat.read_file(filnamn, make_complex=make_complex,
                                      property_to_vars=property_to_vars,
                                      guess_unit=guess_unit,
                                      normalize=normalize,
                                      make_matrix=make_matrix,
                                      merge=merge,
                                      verbose=verbose,
                                      hyper=hyper,
                                      encoding=encoding)


if __name__ == "__main__":
    data = read_spdata("tests/testdata/sp-data/sp_oneport_1_1.txt",
                       merge=False)
    data2 = read_spdata("tests/testdata/sp-data/sp_twoport_1.txt")




