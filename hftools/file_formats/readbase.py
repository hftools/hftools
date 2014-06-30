# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
from __future__ import print_function
import io
import itertools
import re

from itertools import chain
from collections import namedtuple

import numpy as np

import hftools.py3compat as py3
from hftools.constants import unit_to_multiplier
from hftools.dataset import DataBlock, DimPartial, hfarray, make_matrix
from hftools.file_formats.common import normalize_names
from hftools.file_formats import merge_blocks
from hftools.utils import glob
from hftools import path

reg_outerfun = re.compile(r"^([A-Za-z_][A-Za-z_0-9]*)\((.*)\)(.*)")


class FileFormatError(Exception):
    pass


class ParseError(Exception):
    pass

Token = namedtuple("Token", 'tag lineno rad')


class Stream(object):
    def __init__(self, stream):
        self.stream = chain(stream)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.stream)

    def next(self):
        return next(self.stream)

    def push(self, token):
        self.stream = chain([token], self.stream)


def ExpectEndOfFile(stream):
    try:
        token = next(stream)
        raise ParseError("Got token %r when expecting end of file" % (token, ))
    except StopIteration:
        pass


def ManyOptional(name, func=None):
    if func is None:
        func = lambda x: x

    def ManyOptional(stream):
        tokens = []
        current_token = next(stream)
        while current_token.tag == name:
            tokens.append(func(current_token.rad))
            try:
                current_token = next(stream)
            except StopIteration:
                return tokens
        stream.push(current_token)
        return tokens
    return ManyOptional


def Optional(name, func=None):
    if func is None:
        func = lambda x: x

    def Optional(stream):
        try:
            current_token = next(stream)
        except StopIteration:
            return None
        if current_token.tag == name:
            return func(current_token.rad)
        else:
            stream.push(current_token)
    return Optional


def OneOf(names, error, func=None):
    if func is None:
        func = lambda x: x

    def One(stream):
        tokens = []
        current_token = next(stream)
        if current_token.tag in names:
            tokens.append(current_token.rad)
        else:
            raise Exception(error)
        return tokens
    return One


def One(name, error, func=None):
    return OneOf([name], error, func)


def get_outer_function(varname):
    """Return function name and wrapped varname
    """
    m = reg_outerfun.match(varname)
    if m:
        return m.groups() + (varname,)
    else:
        return "", varname, "", varname


def make_cplx(dataset):
    """Inplace conversion in *dataset* of pairs of variables into complex
    variables. Using the pattern Re(x) & Im(x) -> x, and Re(x) and Im(x)
    removed from *dataset*.
    """
    varnames = {}
    for vname in dataset.allvarnames:
        outerfunction, varname, rest, fullname = get_outer_function(vname)
        varnames.setdefault(varname, {})[outerfunction] = (outerfunction,
                                                           rest, fullname)

    for vname, alt in varnames.items():
        if "Re" in alt and "Im" in alt:
            if vname in dataset:
                pass  # Already scalar data present
            else:
                dataset[vname] = (dataset[alt["Re"][-1]] +
                                  dataset[alt["Im"][-1]] * 1j)
                del dataset[alt["Re"][-1]]
                del dataset[alt["Im"][-1]]
        elif "Mag" in alt and "Arg" in alt:
            if vname in dataset:
                pass  # Already scalar data present
            else:
                dataset[vname] = (dataset[alt["Mag"][-1]] *
                                  np.exp(np.pi / 180 * 1j *
                                         dataset[alt["Arg"][-1]]))
                del dataset[alt["Mag"][-1]]
                del dataset[alt["Arg"][-1]]

    return dataset

reg_varunit = re.compile("(.*?)[ \t]*[[]([a-zA-Z]+)[]]*$")


def fix_unit(db, varname):
    res = reg_varunit.match(varname)
    if res:
        vname, unit = res.groups()
        mul, unit = unit_to_multiplier(unit)
#        db.rename(varname, vname)
        if varname in db.ivardata:
            olddim = db.ivardata[varname]
            newdim = olddim.__class__(olddim, name=vname,
                                      data=olddim.data * mul, unit=unit)
            db.replace_dim(olddim, newdim)
        else:
            db.rename(varname, vname)
            db[vname] = db[vname] * mul
            db[vname].unit = unit


def combine_elements_to_matrix(data, name):
    reg = re.compile("%s([0-9])([0-9])" % name)
    i_idx = set()
    j_idx = set()
    vnames = []
    for v in data.vardata:
        res = reg.match(v)
        if res:
            i_idx.add(int(res.groups()[0]))
            j_idx.add(int(res.groups()[1]))
            vnames.append(v)
    if vnames and len(i_idx) * len(j_idx) == len(vnames):
        i_indices = sorted(i_idx)
        j_indices = sorted(j_idx)
        ex = data["%s%d%d" % (name, list(i_idx)[0], list(j_idx)[0])]
        shape = ex.shape
        x = np.empty(shape + (len(i_indices), len(j_indices), ),
                     dtype=ex.dtype)

        for i, j in itertools.product(i_indices, j_indices):
            val = data["%s%d%d" % (name, i, j)]
            x[..., i_indices.index(i), j_indices.index(j), ] = val

    else:
        msg = "Can not build complete matrix using elements with name %r"
        raise Exception(msg % name)

    data[name] = make_matrix(np.array(x), dims=ex.dims)
    return data

reg_matrix = re.compile("([A-Za-z_]+)([0-9])([0-9])$")


def match_matrix_elements(varnames, elem_matcher):
    possibles = {}
    for elemname in varnames:
        res = elem_matcher(elemname)
        if res:
            vname, i, j = res.groups()
            i = int(i)
            j = int(j)
            possibles.setdefault(vname, []).append((i, j, elemname))
    out = {}
    for vname, elem_tuples in possibles.items():
        elems, i, j = _matrix_elements(elem_tuples)
        if elems is not None and len(elems) > 1:
            out[vname] = elems, i, j
    return out


def _matrix_elements(elem_tuples):
    i_idx = set()
    j_idx = set()
    elems = []
    for i, j, elemname in sorted(elem_tuples):
        i_idx.add(i)
        j_idx.add(j)
        elems.append(elemname)

    if len(i_idx) * len(j_idx) != len(elems):
        return None, None, None
    else:
        return elems, sorted(i_idx), sorted(j_idx)


class ReadFileFormat(object):
    def __init__(self, make_complex=True, property_to_vars=True,
                 guess_unit=True, normalize=True, make_matrix=True,
                 merge=True, verbose=False, hyper=False, **kw):
        """class to handle file reading of a datafile
        """

        self.make_complex = make_complex
        self.property_to_vars = property_to_vars
        self.guess_unit = guess_unit
        self.normalize = normalize
        self.make_matrix = make_matrix
        self.merge = merge
        self.verbose = verbose
        self.file_index = 0
        self.hyper = hyper
        for name in kw:
            if not hasattr(self, name):
                setattr(self, name, kw[name])

    @classmethod
    def read_file(cls, filename, make_complex=True, property_to_vars=True,
                  guess_unit=True, normalize=True, make_matrix=False,
                  merge=True, verbose=False, multiple_files=True,
                  hyper=False, encoding="cp1252", **kw):
        obj = cls(make_complex=make_complex, property_to_vars=property_to_vars,
                  guess_unit=guess_unit, normalize=normalize,
                  make_matrix=make_matrix, merge=merge, verbose=verbose,
                  hyper=hyper, **kw)
        obj.filename = filename
        if multiple_files:
            if isinstance(filename, (list, tuple)):
                filenames = []
                for f in filename:
                    filenames.extend(glob(f))
            else:
                filenames = glob(filename)
        else:
            filenames = [filename]
        filenames = [path(f) for f in filenames]
        objs = []
        if not filenames:
            raise IOError("Pattern %r did not match any files" % filename)
        for idx, fname in enumerate(filenames):
            if verbose:
                print("\r%-80s\r" % fname.basename(), end="")
            with io.open(fname, encoding=encoding) as fil:
                res = obj.do_file(fil)
                if multiple_files:
                    #res["FILEINDEX"] = DimPartial("FILEINDEX", [idx])
                    fname = py3.cast_unicode(fname)
                    if "FILENAME" not in res:
                        res["FILENAME"] = hfarray(fname)
                objs.append(res)
        if multiple_files:
            res = obj._merge(objs)
        else:
            res = objs
        if verbose:
            print("\r%80s\r" % "")
        return res

    def do_file(self, stream):
        tokenstream = Stream(self.tokenize(stream))
        stream_of_block_tokens = self.group_blocks(tokenstream)
        blocks = list(self.parse_blocks(stream_of_block_tokens))
        if not blocks:
            args = "No blocks found in file. Perhaps file is empty"\
                   " or of the wrong kind. (Tried to parse %s)"
            msg = args % (self.__class__.__name__)
            raise ParseError(msg)
        blocks = self._make_complex(blocks)
        blocks = self._properties_to_vars(blocks)
        blocks = self._guess_unit(blocks)
        blocks = self._normalize(blocks)
        blocks = self._combine_matrices(blocks)
        #import pdb;pdb.set_trace()
        blocks = self._merge(blocks)
        #import pdb;pdb.set_trace()
        if isinstance(blocks, DataBlock):
            if "FILEINDEX" not in blocks:
                blocks["FILEINDEX"] = DimPartial("FILEINDEX",
                                                 [self.file_index])
        else:
            for b in blocks.values():
                if isinstance(b, (list, tuple)):
                    for x in b:
                        if "FILEINDEX" not in x:
                            x["FILEINDEX"] = DimPartial("FILEINDEX",
                                                        [self.file_index])
                else:
                    if "FILEINDEX" not in b:
                        b["FILEINDEX"] = DimPartial("FILEINDEX",
                                                    [self.file_index])
        self.file_index += 1
        return blocks

    def _make_complex(self, blocks):
        if self.make_complex:
            return [self.make_block_complex(db) for db in blocks]
        else:
            return blocks

    def _properties_to_vars(self, blocks):
        if self.property_to_vars:
            for db in blocks:
                db.values_from_property()
        return blocks

    def _guess_unit(self, blocks):
        if self.guess_unit:
            for db in blocks:
                db.guess_units()
                for v in db.allvarnames:
                    fix_unit(db, v)
        return blocks

    def _normalize(self, blocks):
        out = []
        if self.normalize:
            for db in blocks:
                out.append(self.normalize_names(db))
        else:
            out = blocks
        return out

    def _combine_matrices(self, blocks):
        if not self.make_matrix:
            return blocks
        for block in blocks:
            self._make_matrix_from_block(block)
        return blocks

    def _make_matrix_from_block(self, block):
        matrices = {}
        for matcher in self.match_matrix_element_name:
            matrices.update(match_matrix_elements(block.vardata.keys(),
                                                  matcher))
        for vname, (elems, i_indices, j_indices) in matrices.items():
            ex = block[elems[0]]
            shape = ex.shape
            x = np.empty(shape + (len(i_indices) * len(j_indices),),
                         dtype=ex.dtype)
            for idx, elem in enumerate(elems):
                x[..., idx] = block[elem]
                del block[elem]
            x.shape = shape + (len(i_indices), len(j_indices),)
            block[vname] = make_matrix(x, dims=ex.dims)

    def _merge(self, blocks):
        if self.merge:
            if len(blocks) == 1:
                return blocks[0]
            else:
                return merge_blocks(blocks, hyper=self.hyper)
        else:
            return blocks

    def tokenize(self, stream):
        pass

    def group_blocks(self, stream):
        """Default for files that do not have multiple groups.
        """
        return stream

    def parse_blocks(self, stream):
        pass

    def make_block_complex(self, block):
        return make_cplx(block)

    def normalize_names(self, block):
        return normalize_names(block)

    def id_matrix_elements(self, block):
        pass

    def combine_blocks(self, blocks):
        pass

    reg = re.compile("([A-Za-z_]+)([0-9])([0-9])$")
    match_matrix_element_name = [reg.match]
    del reg
