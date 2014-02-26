# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

u"""
Touchstone
==========


    .. autofunction:: read_touchstone
    .. autofunction:: save_touchstone

"""
import re
import numpy as np
from numpy import iscomplexobj, sqrt

from hftools.math import dB_angle_to_complex, mag_angle_to_complex,\
    re_im_to_complex
from hftools.dataset import DataBlock, DimSweep, hfarray,\
    make_matrix, DimMatrix_i, DimMatrix_j
from hftools.file_formats.common import Comments, db_iterator,\
    make_col_from_matrix
from hftools.file_formats.readbase import Token
from hftools.file_formats.readbase import ReadFileFormat


class TouchstoneError(Exception):
    pass


class ReadTouchstoneFileFormat(ReadFileFormat):

    def tokenize(self, stream):
        for idx, rad in enumerate(stream):
            lineno = idx + 1
            rad = rad.strip()
            if not rad:  # skip empty lines
                continue
            elif rad.startswith("!"):  # Comment line with information
                yield Token("Comments", lineno, rad[1:].strip())
            elif rad.startswith("#"):
                if is_touchstone(None, rad):
                    yield Token("Info", lineno, rad[1:].strip())
                else:
                    msg = "# format is invalid on line: %s" % lineno
                    raise TouchstoneError(msg)
            else:
                yield Token("Data", lineno, rad.split("!", 1)[0].strip())

    def parse_blocks(self, stream):
        comments = []
        info = None
        datalist = []
        for token, lineno, rad in stream:
            if token == "Comments":
                comments.append(rad)
            elif token == "Info":
                if info is None:
                    info = rad
                else:
                    msg = "Second # info at lineno: %d" % lineno
                    raise TouchstoneError(msg)
            else:
                datalist.append(map(float, rad.split()))
        if info is None:
            raise TouchstoneError("No # info line in file")
        comments = Comments(comments)
        f, data, fn, noisedata = proc_data(datalist)
        out = proc_info(info, f, data, fn, noisedata)
        #HZ S RI R 50
        #comments.add_from_comment("!INFO:#%s"%(" ".join(info)))
        out.comments = comments
        yield out


def process_data(data):
    max_N = int(sqrt(sum(map(len, data)) - 1) / 2 + 1)
    n = 1  # number of ports
    for n in range(1, max_N):
        N = 2 * n ** 2 + 1  # number of data points per frequency
        output, noise = _process_data(data, N)
        if output:
            return output, noise
    raise TouchstoneError("File is not a valid Touchstone File")


def _process_data(data, N):
    output = []
    current_row = []
    noise = []
    for idx, rad in enumerate(data):
        if not current_row:
            if output:
                if rad[0] <= output[-1][0]:
                    if N == 9:  # N==9 when two-port
                        noise, _ = _process_data(data[idx:], 5)
                        if noise:
                            return output, noise
                    return None, None
                else:
                    current_row.extend(rad)
            else:
                current_row.extend(rad)
        else:
            current_row.extend(rad)

        if len(current_row) > N:
            return None, None
        elif len(current_row) == N:
            output.append(current_row)
            current_row = []
    if current_row:
        return None, None
    return output, noise


def proc_data(data, cplxtype=None):
    output, noise = process_data(data)
    data = np.array(output)
    f = np.array(data[:, 0])
    s = data[:, 1:]
    if noise:
        data = np.array(noise)
        fn = data[:, 0]
        noisedata = np.array(data[:, 1:], dtype=np.float64)
        return f, s, fn, noisedata
    else:
        return f, s, None, None


reg_touch = re.compile(r"\s*#\s*[kmgtp]?hz\s+[szygh]\s+" +
                       r"([a-z][a-z])(\s+r\s+[0-9]+)?", re.I)


def is_touchstone(filename, rad):
    return reg_touch.match(rad)


#The info string is often capitalized and so can not distinguish between
#upper and lowcase, thus we assume only prefixes >1
freqmultdict = dict(HZ=1, KHZ=1e3, MHZ=1e6, GHZ=1e9, THZ=1e12, PHZ=1e15)


def proc_info(info, f, data, fn, noisedata):
    info_list = info.upper().split()
    if len(info_list) == 5:
        info = freq_mult, twoporttype, datatype, rtype, z0 = info_list
    elif len(info_list) == 3:
        info = freq_mult, twoporttype, datatype = info_list
        z0 = 1.
    else:
        msg = ("# format should only have 5 values this one has %d" %
               len(info_list))
        raise TouchstoneError(msg)
    z0 = float(z0)
    f = f * freqmultdict[freq_mult]
    if fn is not None:
        fn = fn * freqmultdict[freq_mult]

    convfunc = dict(RI=re_im_to_complex,
                    DB=dB_angle_to_complex,
                    MA=mag_angle_to_complex)

    if datatype not in convfunc:
        pars = datatype
        msg = "Unknown dataformat: %s, valid formats: RI, DB, MAG" % pars
        raise TouchstoneError(msg)

    out = np.empty((data.shape[0], data.shape[1] // 2), dtype=np.complex128)

    out[...] = convfunc[datatype](data[:, ::2],  data[:, 1::2])
    if out.shape[1] == 1:
        out.shape = (out.shape[0], 1, 1)
    elif out.shape[1] == 4:
        out.shape = (out.shape[0], 2, 2)
        out = out.transpose(0, 2, 1)
    else:
        N = int(sqrt(out.shape[1]))
        out.shape = (out.shape[0], N, N)

    dims = (DimSweep("freq", f, unit="Hz"), )
    if fn is not None and len(f) == len(fn) and np.allclose(f, fn):
        noisedims = dims
    elif fn is None:
        noisedims = None
    else:
        noisedims = (DimSweep("freqn", fn, unit="Hz"), )
    out = make_matrix(out, dims)
    f = hfarray(f, dims)
    db = DataBlock()
    db[info[1]] = out
    if noisedata is not None:
        db.Rn = hfarray(noisedata[..., -1], dims=noisedims) * z0
        db.Fmin = 10 ** (hfarray(noisedata[..., 0], dims=noisedims) / 10)
        db.Gopt = hfarray(convfunc[datatype](noisedata[:, 1], noisedata[:, 2]),
                          dims=noisedims)
    db.Z0 = hfarray(z0, unit="Ohm")
    return db


def format_touchstone_block(sweepvars, header, fmts,
                            columns, blockname, comments):
    for comment in comments.fullcomments:
        yield ["!" + comment]
    yield [comments.property.get("INFO", ["#HZ S RI R 50"])[0]]
    header, columns = make_col_from_matrix(header, columns,
                                           "%s[%s,%s]", fortranorder=True)
    fmts = [x.outputformat for x in columns]
    for row in zip(*columns):
        out = []
        for elem, fmt in zip(row, fmts):
            if iscomplexobj(elem):
                out.append(fmt % elem.real)
                out.append(fmt % elem.imag)
            else:
                out.append(fmt % elem)
        yield out


def save_touchstone(db, filename):
    """Write a Datablock to a touchstone-format file with name filename.
    """
    fil = open(filename, "w")
    for rad in db_iterator(db, format_touchstone_block):
        fil.write("\t".join(rad))
        fil.write("\n")


def read_touchstone(filnamn, make_complex=True, property_to_vars=True,
                    guess_unit=True, normalize=True, make_matrix=True,
                    merge=True, verbose=False):
    res = ReadTouchstoneFileFormat.read_file(filnamn,
                                             make_complex=make_complex,
                                             property_to_vars=property_to_vars,
                                             guess_unit=guess_unit,
                                             normalize=normalize,
                                             make_matrix=make_matrix,
                                             merge=merge,
                                             verbose=verbose)
    return res

if __name__ == "__main__":
    a = read_touchstone("tests/testdata/touchstone/test4.s2p",
                        make_complex=True,
                        property_to_vars=True,
                        guess_unit=True,
                        normalize=True,
                        make_matrix=True,
                        merge=True,
                        verbose=True)

    d = DataBlock()
    d.comments = Comments([])
    fi = DimSweep("freq", [0e9, 1e9, 2e9])
    d.S = hfarray([[[1 + 1j, 1 + 2j], [2 + 1j, 2 + 2j]]] * 3,
                  dims=(fi, DimMatrix_i("i", 2), DimMatrix_j("j", 2),))
    save_touchstone(d, "tests/testdata/touchstone/savetest/res_1.txt")

