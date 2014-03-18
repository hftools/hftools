# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

u"""
MDIF
====


    .. autoclass:: MDIFBunch

"""
from __future__ import print_function
import re
import itertools
import time
import numpy as np

from hftools.dataset import DataBlock, DataDict, hfarray, DimSweep,\
    DimRep, DimPartial, make_matrix,\
    make_vector, yield_dim_consistent_datablocks, convert_matrices_to_elements


from hftools.utils import glob
from hftools.file_formats.common import Comments,\
    format_complex_header, format_elem
from hftools.file_formats.readbase import ManyOptional,\
    One, Token, ReadFileFormat, FileFormatError


class MDIFError(FileFormatError):
    pass

reg_mdif = re.compile(r"[\t ]*VAR[\t ]*[A-Za-z_0-9.]+[(][a-zA-Z]+[)]")


def is_mdif(filename, rad):
    return bool(reg_mdif.match(rad))


reg = re.compile("(.+)[(](integer|real|string|complex)[)]")


class GetData(object):
    def __init__(self, header):
        self.header = header
        self.parse_stream = []
        for headline in header:
            self.parse_stream.append([])
            for head in headline:
                self.parse_stream[-1].append(self.parse_header(head))

    def sweep_name(self):
        return self.parse_stream[0][0][1]

    def parse_data(self, datastream):
        vardata = DataDict()
        while True:
            for lineparser in self.parse_stream:
                linedata = datastream.pop(0).split()
                for func, varname in lineparser:
                    vardata.setdefault(varname, []).append(func(linedata))
            if not datastream:
                break
        return vardata

    def parse_header(self, head):
        res = reg.match(head)
        if res:
            var, vtype = res.groups()
            typefunc = {"string": self.get_string,
                        "complex": self.get_complex,
                        "real": self.get_real,
                        "integer": self.get_integer}[vtype]
        else:
            raise Exception("Unknown datatype in header %r" % head)
        return typefunc, var

    def get_string(self, stream):
        d = stream.pop(0)
        d = d.strip('"')
        return d

    def get_real(self, stream):
        d = stream.pop(0)
        d = d.strip('"')
        d = float(d)
        return d

    def get_integer(self, stream):
        d = stream.pop(0)
        d = d.strip('"')
        d = int(d)
        return(d)

    def get_complex(self, stream):
        dr = self.get_real(stream)
        di = self.get_real(stream)
        d = dr + 1j * di
        return d


class ReadMDIFFileFormat(ReadFileFormat):

    def __init__(self, *k, **kw):
        """
        """
        ReadFileFormat.__init__(self, *k, **kw)

    @classmethod
    def read_file(cls, filename, make_complex=True, property_to_vars=True,
                  guess_unit=True, normalize=True, make_matrix=False,
                  merge=True, verbose=False, multiple_files=True,
                  blockname=None, **kw):

        if multiple_files:
            merge = False
        obj = cls(make_complex=make_complex, property_to_vars=property_to_vars,
                  guess_unit=guess_unit, normalize=normalize,
                  make_matrix=make_matrix, merge=merge, verbose=verbose, **kw)
        obj.filename = filename

        if multiple_files:
            filenames = glob(filename)
        else:
            filenames = [filename]
        objs = {}
        for idx, fname in enumerate(filenames):
            with open(fname) as fil:
                res = obj.do_file(fil)
                for k, v in res.iteritems():
                    if isinstance(v, list):
                        for a in v:
                            a["FILENAME"] = hfarray(fname)
                        objs.setdefault(k, []).extend(v)
                    else:
                        v["FILENAME"] = hfarray(fname)
                        objs.setdefault(k, []).append(v)

        if multiple_files:
            res = {}
            for k, v in objs.iteritems():
                res[k] = simple_merge_blocks(v)
                res[k].guess_units()
        else:
            res = objs

        if blockname:
            if blockname in res:
                return res[blockname]
            else:
                msg = "%r not a block in mdif file %r" % (blockname, filename)
                MDIFError(msg)
        else:
            return res

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
        for idx, line in enumerate(stream):
            lineno = idx + 1
            if self.verbose:
                print("\r               \r", lineno, end="")
            line = line.lstrip()
            if not line:
                continue
            elif line[:1] == "!":
                yield Token("COMMENT", lineno, line[1:].strip())
            elif line[:1] == "%":
                yield Token("HEADER", lineno, line[1:].strip())
            elif line[:1] == "#":
                yield Token("ATTRIB", lineno, line[1:].strip())
            elif line.startswith("BEGIN"):
                yield Token("BEGIN", lineno, line[6:].strip())
            elif line.startswith("END"):
                yield Token("END", lineno, line[4:])
            elif line.startswith("VAR"):
                yield Token("VAR", lineno, line[4:])
            else:
                yield Token("DATA", lineno, line)

    def group_blocks(self, stream):
        """Bunch tagged stream into sub blocks. Each sub block
        containing a measurement data block.
        """
        while True:
            comments = ManyOptional("COMMENT")(stream)

            def split_eq(rad):
                return [x.strip() for x in rad.split("=")]
            var = ManyOptional("VAR", split_eq)(stream)
            blockname = One("BEGIN", "Must have Begin after VARs")(stream)
            attribs = ManyOptional("ATTRIB", split_eq)(stream)
            header = ManyOptional("HEADER", lambda rad: rad.split())(stream)
            data = ManyOptional("DATA")(stream)
            out = (comments, var, blockname, header, attribs, data)
            One("END", "Must have END after DATA")(stream)
            yield out

    def parse_blocks(self, stream):
        for comments, vars, blockname, header, attribs, data in stream:
            db = DataBlock()
            db.blockname = blockname = blockname[0]
            #self.header = header = header[0]
            db.comments = Comments(comments)
            #data = numpy.array(data)
            dd = self.proc_data(header, data[:])
            for vname in dd:
                db[vname] = dd[vname]
            for var in vars:
                vi = self.proc_var(var)
                db[vi.name] = vi
            for var in attribs:
                name, value = self.proc_attrib(var)
                db[name] = value

            self.make_matrices(db, sum(header, []))
            yield db

    def proc_attrib(self, var):
        varname, value = var
        varname, vartype = varname.split('(')
        vartype = vartype.strip(")")
        if varname.startswith("SWEEP."):
            varname = varname[len("sweep."):]
        if vartype == "real":
            value = float(value)
        else:
            raise Exception("%r not an implemented var type" % vartype)
        return varname, hfarray(value)

    def proc_var(self, var):
        varname, value = var
        varname, vartype = varname.split('(')
        vartype = vartype.strip(")")
        if varname.startswith("SWEEP."):
            varname = varname[len("sweep."):]
        if vartype == "real":
            value = float(value)
        else:
            raise Exception("%r not an implemented var type" % vartype)
        i = DimPartial(varname, np.array([value]))
        return i

    def proc_data(self, header, data):
        processor = GetData(header)
        vardata = processor.parse_data(data)

        oldsweepname = processor.sweep_name()
        sweepname = processor.sweep_name()
        if sweepname.lower().startswith("sweep."):
            sweepname = sweepname[len("sweep."):]

        fi = DimSweep(sweepname, vardata[oldsweepname])
        del vardata[oldsweepname]
        for vname in vardata:
            vardata[vname] = hfarray(vardata[vname], (fi,))
        return vardata

    def make_matrices(self, db, header):
        matrices, vectors, scalars = match_matrix_names(header)
        for matrix, value in matrices.iteritems():
            a = [(i, j) for x, i, j, y in value]
            a.sort()
            arrays = []
            for i, j in a:
                vname = "%s[%s,%s]" % (matrix, i, j)
                arrays.append(np.array(db.vardata[vname])[:, np.newaxis])
            dims = db.vardata["%s[%s,%s]" % (matrix, i, j)].dims
            X = np.concatenate(arrays, 1)
            I, J = map(set, zip(*a))
            X.shape = X.shape[:1] + (len(I), len(J))
            db.vardata[matrix] = make_matrix(X, dims)
            idx = min([db.vardata.order.index("%s[%s,%s]" % (matrix, i, j))
                       for i, j in a])
            for i, j in a:
                del db.vardata["%s[%s,%s]" % (matrix, i, j)]
            del db.vardata.order[db.vardata.order.index(matrix)]
            db.vardata.order.insert(idx, matrix)

        for matrix, value in vectors.iteritems():
            a = [(i,) for x, i, y in value]
            a.sort()
            fmt = "%s[%%s]" % (matrix, )
            arrays = [np.array(db.vardata[fmt % i])[:, np.newaxis] for i, in a]
            dims = db.vardata["%s[%s]" % (matrix, i)].dims
            X = np.concatenate(arrays, 1)
            X.shape = X.shape[:1] + (len(a),)
            db.vardata[matrix] = make_vector(X, dims)
            idx = min([db.vardata.order.index("%s[%s]" % (matrix, i))
                       for i, in a])
            for i, in a:
                del db.vardata["%s[%s]" % (matrix, i)]
            del db.vardata.order[db.vardata.order.index(matrix)]
            db.vardata.order.insert(idx, matrix)

    def _merge(self, blocks):
        if self.merge:
            block_groups = {}
            for block in blocks:
                block_groups.setdefault(block.blockname, []).append(block)
            out = {}
            for bname in block_groups:
                out[bname] = simple_merge_blocks(block_groups[bname])
                out[bname].blockname = bname
        else:
            out = {}
            for block in blocks:
                out.setdefault(block.blockname, []).append(block)
        return out


def simple_merge_blocks(blocks):
    if len(blocks) == 1:
        return blocks[0]
    ri = DimRep("INDEX", len(blocks))
    out = DataBlock()
    #import pdb;pdb.set_trace()
    for k in blocks[0].vardata:
        out[k] = hfarray([b[k] for b in blocks],
                         dims=(ri,) + blocks[0][k].dims)
        if blocks[0][k].dims:
            out[k] = out[k].reorder_dimensions(blocks[0][k].dims[0])
    for k in blocks[0].ivardata:
        if isinstance(blocks[0].ivardata[k], DimPartial):
            out[k] = hfarray([b.ivardata[k].data[0] for b in blocks],
                             dims=(ri,))
    c = []
    for b in blocks:
        if b.comments is not None:
            c.extend(b.comments.fullcomments)
    out.comments = Comments()
    out.comments.fullcomments = c
    return out


class MDIFBunch:
    """
    bunch = (ManyOptional("COMMENT"),
             ManyOptional("VAR"),
             One("BEGIN"),
             ManyOptional("ATTRIB"),
             One("HEADER"),
             ManyOptional("DATA"),
             One("END"))
    """


def make_block_args(out):
    """
    """
    comments = []
    blockname = None
    for (comments, vars, blockname, headers, attribs, data) in out:
        yield (comments, vars, blockname[0], headers[0], attribs, data)


class MDIFParseError(Exception):
    pass


def match_matrix_names(header):
    reg_matrix = re.compile(r"^([^[]+)(\[(\d+)(,(\d+))?\])?"
                            r"\(([a-zA-Z0-9]+)\)$")
    scalar = []
    matrix = {}
    vector = {}
    offset = 0
    for v in header:
        m = reg_matrix.match(v)
        if not m:
            print("%s not a valid variable" % v)
        else:
            vname, x, i, matrixp, j, vtype = m.groups()
            if x and matrixp:
                tupledata = (offset, int(i), int(j), vtype)
                matrix.setdefault(vname, []).append(tupledata)
            elif x:
                tupledata = (offset, int(i), vtype)
                vector.setdefault(vname, []).append(tupledata)
            else:
                scalar.append((vname, offset, vtype))
            offset += dict(complex=2).get(vtype.lower(), 1)

    for key in list(matrix.keys()):
        value = matrix[key]
        if is_matrix_name(matrix[key]):
            pass
        else:
            for (offset, i, j, vtype) in value:
                scalar.append((key, offset, vtype))
            del matrix[key]

    for key in list(vector.keys()):
        value = vector[key]
        if is_vector_name(vector[key]):
            pass
        else:
            for (offset, i, vtype) in value:
                scalar.append((key, offset, vtype))
            del vector[key]
    return matrix, vector, scalar


def is_matrix_name(lista):
    a = [(i, j) for x, i, j, y in lista]
    I, J = map(set, zip(*a))

    if ((len(I) * len(J) == len(lista) and
         I == set(range(1, len(I) + 1)) and
         J == set(range(1, len(J) + 1)))):
        return True
    else:
        return False
    return a


def is_vector_name(lista):
    a = [i for x, i, y in lista]
    I = set(a)
    proper_indices = set(range(1, len(a) + 1))
    return I == proper_indices


def shorten_name(vdict, name):
    names = name.split(".")
    if len(names) <= 1:
        return name
    for i in range(len(names)):
        newname = ".".join(names[-(i + 1):])
        if newname not in vdict:
            vdict.rename(name, newname)
            return newname
    return name


def fixnames(mdif):
    for b in mdif.blocks.values():
        for v in list(b.vardata):
            if v.endswith(".s"):
                del b.vardata[v]
                continue
            elif v.endswith(".m"):
                b.vardata.rename(v, v[:-2])
                v = v[:-2]
            v = shorten_name(b.vardata, v)


def fmt_mdif_block(dims, db):
    dimnames = [x.name for x in dims]
    fmts = [x.outputformat for x in dims]
    for coord in itertools.product(*[x.data for x in dims[1:]]):
        for iname, fmt, value in zip(dimnames[1:], fmts[1:], coord):
#            import pdb;pdb.set_trace()
            yield [("VAR SWEEP.%s(real) = " + fmt) % (iname, value)]
        blockname = db.blockname
        if blockname is None:
            blockname = "DATABLOCK"
        yield ["BEGIN %s" % blockname]

        attribnames, attribdata = zip(*[(k, v) for (k, v) in db.vardata.items()
                                        if v.dims == dims[1:]])
        attribheader = format_complex_header(attribnames, attribdata,
                                             "%s(real)", "%s(complex)",
                                             None, unit_fmt=False)
        for name, data in zip(attribheader, attribdata):
            yield [("# %s = %s" % (name, data.outputformat)) % data[coord]]

        header, columns = zip(*[(k, v) for (k, v) in db.vardata.items()

                                if v.dims == dims])
        header = (dims[0].name, ) + header

        idx = (Ellipsis,) + coord
        columns = [col[idx] for col in columns]
        columns = [hfarray(dims[0])] + columns
        outheader = format_complex_header(header, columns,
                                          "%s(real)", "%s(complex)",
                                          None, unit_fmt=False)

        yield ["% " + " ".join(outheader)]
        fmts = [x.outputformat for x in columns]
#        import pdb;pdb.set_trace()
        for row in zip(*columns):
            out = []
            for elem, fmt in zip(row, fmts):
                out.extend(format_elem(fmt, elem))
            yield out
        yield ["END"]


def loop_db(db):
    for dims, subset in yield_dim_consistent_datablocks(db):
        if not dims:
            continue
        for rad in fmt_mdif_block(dims, subset):
            yield rad


def save_mdif(db, filename):
    fil = open(filename, "w")
    fil.write("!Created: %s\n" % time.asctime())
    #for rad in db_iterator(db, format_mdif_block):
    for rad in loop_db(convert_matrices_to_elements(db)):
        fil.write("\t".join(rad))
        fil.write("\n")
    fil.close()


def read_mdif(filnamn, make_complex=True, property_to_vars=True,
              guess_unit=True, normalize=True, make_matrix=True,
              merge=True, blockname=None, verbose=True, multiple_files=True):
    return ReadMDIFFileFormat.read_file(filnamn, make_complex=make_complex,
                                        property_to_vars=property_to_vars,
                                        guess_unit=guess_unit,
                                        normalize=normalize,
                                        make_matrix=make_matrix,
                                        merge=merge,
                                        blockname=blockname,
                                        verbose=verbose,
                                        multiple_files=multiple_files)


if __name__ == "__main__":
    mdif = read_mdif("./tests/testdata/mdif/small.mdif", make_complex=True,
                     property_to_vars=True,
                     guess_unit=True, normalize=True, make_matrix=True,
                     merge=True)
    mdif74 = read_mdif("./tests/testdata/mdif/A74.mdif", make_complex=True,
                       property_to_vars=True,
                       guess_unit=True,
                       normalize=True,
                       make_matrix=True,
                       merge=True)["A74.SP"][0]
    f1 = "./tests/testdata/mdif/test1.mdif"
    mdif1 = read_mdif("./tests/testdata/mdif/test1.mdif", make_complex=True,
                      property_to_vars=True, guess_unit=True, normalize=True,
                      make_matrix=True, merge=True)

