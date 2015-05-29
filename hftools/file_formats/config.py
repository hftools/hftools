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
import datetime
import re
import warnings

import yaml

from collections import OrderedDict
from hftools.constants import siprefixes
from hftools import path


def convert_si_value(value):
    try:
        res = reg_si_value.match(value)
    except TypeError:
        return value, None
    if res:
        num, _, fullunit, exp, unit = res.groups()
        if exp is None:
            exp = ""
        return float(num) * siprefixes[exp], unit
    else:
        return value, None


class LexicalScopeList(list):
    def __init__(self, *k):
        list.__init__(self, *k)
        self.parent = None

    def convert_to_standard(self):
        out = []
        for v in self:
            if hasattr(v, "convert_to_standard"):
                out.append(v.convert_to_standard())
            else:
                out.append(v)
        return out


def find_next_dict(dct):
    try:
        parent = dct.parent
    except AttributeError:
        return None
    if isinstance(parent, dict):
        return parent
    else:
        return find_next_dict(parent)


class LexicalScopeDict(OrderedDict):
    def __init__(self, *k, **kw):
        self.parent = None
        OrderedDict.__init__(self, *k, **kw)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        if OrderedDict.__contains__(self, key):
            return True
        return self.parent and key in self.parent

    def update(self, other):
        for k, v in other.items():
            self[k] = v

    def __getitem__(self, key):
        if OrderedDict.__contains__(self, key):
            return OrderedDict.__getitem__(self, key)
        elif ((not OrderedDict.__contains__(self, key)) and
              (self.parent is not None)):
            dct = find_next_dict(self)
            if dct is None:
                raise KeyError("%r not in dictionary" % key)
            else:
                return dct[key]
        else:
            raise KeyError("%r not in dictionary" % key)

    def convert_to_standard(self):
        out = dict()
        for k, v in self.items():
            if hasattr(v, "convert_to_standard"):
                out[k] = v.convert_to_standard()
            else:
                out[k] = v
        return out

    def copy(self):
        return LexicalScopeDict(OrderedDict.copy(self))

    def __repr__(self):
        return dict.__repr__(self)


class MergedDicts(object):
    def __init__(self, *dicts):
        self.dicts = dicts[::-1]
        self.parent = None

    def copy(self):
        m = MergedDicts()
        m.dicts = self.dicts[:]
        return m

    def __contains__(self, key):
        for dct in self.dicts:
            if key in dct:
                return True
        if self.parent:
            return key in self.parent
        return False

    def keys(self):
        a = set()
        for dct in self.dicts:
            a.update(dct.keys())
        return a

    def __getitem__(self, key):
        for dct in self.dicts:
            try:
                return dct[key]
            except KeyError:
                pass
        if self.parent:
            return self.parent[key]
        raise KeyError("Unknown key %r" % key)


def ensure_lexical(obj, parent):
    if isinstance(obj, list):
        l = LexicalScopeList(obj, parent)
    elif isinstance(obj, dict):
        l = LexicalScopeDict(obj, parent)
    else:
        l = obj
    return l


def ensure_standard(obj, parent):
    if isinstance(obj, list):
        l = LexicalScopeList(obj, parent)
    elif isinstance(obj, dict):
        l = LexicalScopeDict(obj, parent)
    else:
        l = obj
    return l


class LexicalConfig(LexicalScopeDict):
    def __init__(self, filename):
        LexicalScopeDict.__init__(self)
        self.filename = filename
        self.reload_settings()

    def reload_settings(self):
        self.clear()
        try:
            with io.open(self.filename, encoding="utf-8") as stream:
                self.update(yaml.load(stream, Loader=LexicalLoader))
        except IOError:
            print("missing config file %r" % self.filename)

    def write(self, filename=None):
        if filename is None:
            filename = self.filename
        with io.open(filename, "w", encoding="utf-8") as stream:
            yaml.dump(self.convert_to_standard(), stream,
                      Dumper=StandardDumper,
                      default_flow_style=False)


class Config(dict):
    def __init__(self, filename):
        dict.__init__(self)
        self.filename = filename
        self.reload_settings()

    def reload_settings(self):
        self.clear()
        try:
            with io.open(self.filename, encoding="utf-8") as stream:
                self.update(yaml.load(stream, Loader=StandardLoader))
        except IOError:
            warnings.warn("missing config file %r" % self.filename)

    def write(self, filename=None):
        if filename is None:
            filename = self.filename
        with io.open(filename, "w", encoding="utf-8") as stream:
            yaml.dump(dict(self), stream, Dumper=StandardDumper,
                      default_flow_style=False)


class SIValue(float):
    def __new__(self, value):
        value, unit = convert_si_value(value)
        self.unit = unit
        x = float.__new__(self, value)
        x.unit = unit
        return x


def convert_or_none(x):
    if x is None:
        return x
    else:
        return float(x)


def sivalue_representer(dumper, data):
    if data.unit:
        return dumper.represent_scalar(u'tag:yaml.org,2002:str', u'%s%s' % (data, data.unit))
    else:
        return dumper.represent_scalar(u'tag:yaml.org,2002:float', u'%s' % (data))



def timeintervall_representer(dumper, data):
    tremain = data.total_seconds()
    days = tremain // (3600 * 24)
    tremain = tremain - days * 3600 * 24
    hours = tremain // (3600)
    tremain = tremain - hours * 3600
    minutes = tremain // 60
    tremain = tremain - minutes * 60
    seconds = tremain
    out = []
    if days:
        out.append(u"%s days" % days)
    if hours:
        out.append(u"%s hours" % hours)
    if minutes:
        out.append(u"%s minutes" % minutes)
    if seconds:
        out.append(u"%s seconds" % seconds)

    return dumper.represent_scalar(u'tag:yaml.org,2002:str',
                                   u" ".join(out))


def timestamp_representer(dumper, data):
    return dumper.represent_scalar(u'tag:yaml.org,2002:str', u'%s' % (data))


def path_representer(dumper, p):
    return dumper.represent_scalar(u'!str', u'%s' % p)

reg_si_value = re.compile(r"^([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
                          r"[ \t]*(([dcmunpfakMGTPE])?(Hz|V|A|m|s|W|Ohm)?)?$")

reg_percent_value = re.compile(r"^([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
                               r"[ \t]*(%|ppm|ppb)$")


regtime = re.compile(r"([0-9][0-9]):([0-9][0-9])(:([0-9][0-9]))?")
regtimeshort = re.compile(r"[0-9][0-9]:[0-9][0-9]")
regdatetimeshort = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
                              r" [0-9][0-9]:[0-9][0-9]")
regdate = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

rec = re.compile
timeintervallpattern = rec(r'^(([0-9]+([.][0-9]*)?)[ ]*(?:days|day))?[ ]*'
                           r'(([0-9]+([.][0-9]*)?)[ ]*(?:hour|hours))?[ ]*'
                           r'(([0-9]+([.][0-9]*)?)[ ]*(?:minutes|minute))?[ ]*'
                           r'(([0-9]+([.][0-9]*)?)[ ]*(?:second|seconds))?$')


def sivalue_constructor(loader, node):
    value = loader.construct_scalar(node)
    return SIValue(value)


def percentvalue_constructor(loader, node):
    value = loader.construct_scalar(node)
    res = reg_percent_value.match(value)
    v, _, kind = res.groups()
    if kind == "%":
        return float(v) * 0.01
    elif kind == "ppm":
        return float(v) * 1e-6
    elif kind == "ppb":
        return float(v) * 1e-9
    else:  # pragma: no cover
        raise ValueError("Not a valid percent, ppm, ppb kind %r" % kind)


def timeintervall_constructor(loader, node):
    value = loader.construct_scalar(node)
    res = timeintervallpattern.match(value)
    _, days, _, _, hours, _, _, minutes, _, _, seconds, _ = res.groups()
    days = 0 if days is None else float(days)
    hours = 0 if hours is None else float(hours)
    minutes = 0 if minutes is None else float(minutes)
    seconds = 0 if seconds is None else float(seconds)

    return datetime.timedelta(days=days,
                              hours=hours,
                              minutes=minutes,
                              seconds=seconds)


def shortdatetime_constructor(loader, node):
    value = loader.construct_scalar(node)
    return datetime.datetime.strptime(value, "%Y-%m-%d %H:%M")


def time_constructor(loader, node):
    value = loader.construct_scalar(node)
    res = regtime.match(value)
    h, m, _, s = res.groups()
    if s is None:
        s = 0
    t = datetime.time(int(h), int(m), int(s))
    dnow = datetime.datetime.today()
    d2 = datetime.datetime.combine(dnow, t)
    if d2 < dnow:
        d2 = d2 + datetime.timedelta(days=1)
    return d2


def overrideint(loader, node):
    value = loader.construct_scalar(node)
    res = regtime.match(value)
    if res:
        return time_constructor(loader, node)
    else:
        return loader.construct_yaml_int(node)


class LexicalLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            msg = "expected a mapping node, but found %s" % node.id
            raise yaml.ConstructorError(None, None,
                                        msg,
                                        node.start_mark)
        mapping = LexicalScopeDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                msg = "while constructing a mapping"
                msg2 = "found unacceptable key (%s)" % exc
                raise yaml.ConstructorError(msg, node.start_mark,
                                            msg2, key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            if isinstance(value, (LexicalScopeDict, LexicalScopeList)):
                value.parent = mapping
            mapping[key] = value
        return mapping

    def construct_sequence(self, node, deep=False):
        if not isinstance(node, yaml.SequenceNode):
            msg = "expected a sequence node, but found %s" % node.id
            raise yaml.ConstructorError(None, None, msg, node.start_mark)
        out = LexicalScopeList()
        for child in node.value:
            value = self.construct_object(child, deep=deep)
            if isinstance(value, (LexicalScopeDict, LexicalScopeList)):
                value.parent = out
            out.append(value)
        return out

    def construct_yaml_map(self, node):
        yield self.construct_mapping(node)

    def construct_yaml_seq(self, node):
        yield self.construct_sequence(node)


class StandardLoader(yaml.SafeLoader):
    pass


class StandardDumper(yaml.SafeDumper):
    pass

for loader in [LexicalLoader, StandardLoader]:
    loader.add_constructor(u'!sivalue', sivalue_constructor)
    loader.add_constructor(u'!timeintervall', timeintervall_constructor)
    loader.add_constructor(u'!shortdatetime', shortdatetime_constructor)
    loader.add_constructor(u'!time', time_constructor)
    loader.add_constructor(u'!percentvalue', percentvalue_constructor)
    loader.add_constructor(u'tag:yaml.org,2002:int', overrideint)

LexicalLoader.add_constructor('tag:yaml.org,2002:map',
                              LexicalLoader.construct_yaml_map)
LexicalLoader.add_constructor('tag:yaml.org,2002:seq',
                              LexicalLoader.construct_yaml_seq)


def lexical_scope_dict_representer(dumper, data):
    return dumper.represent_mapping(u'tag:yaml.org,2002:map', data.items())


def lexical_scope_list_representer(dumper, data):
    return dumper.represent_mapping(u'tag:yaml.org,2002:seq', data)


StandardDumper.add_representer(SIValue, sivalue_representer)
StandardDumper.add_representer(datetime.datetime, timestamp_representer)
StandardDumper.add_representer(datetime.timedelta, timeintervall_representer)
StandardDumper.add_representer(LexicalScopeDict,
                               lexical_scope_dict_representer)
StandardDumper.add_representer(LexicalScopeList,
                               lexical_scope_list_representer)
StandardDumper.add_representer(path, path_representer)

for loader_or_dumper in [LexicalLoader, StandardLoader, StandardDumper]:
    loader_or_dumper.add_implicit_resolver(u'!sivalue', reg_si_value, None)
    loader_or_dumper.add_implicit_resolver(u'!percentvalue',
                                           reg_percent_value, None)
    loader_or_dumper.add_implicit_resolver(u'!timeintervall',
                                           timeintervallpattern, None)
    loader_or_dumper.add_implicit_resolver(u'!shortdatetime',
                                           regdatetimeshort, None)
    loader_or_dumper.add_implicit_resolver(u'!time', regtime, None)
