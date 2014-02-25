# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import itertools
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.projections.polar import PolarAxes
from matplotlib.ticker import FormatStrFormatter, EngFormatter
import matplotlib
import numpy as np
import pylab
from matplotlib._pylab_helpers import Gcf
from matplotlib.backends.backend_pdf import PdfPages
import hftools.math
from hftools.constants.si_units import si_exp_to_prefixes, _help_format_sci
from hftools.dataset import hfarray, remove_tail
from hftools.math import delay, unwrap_phase

"""
TODO
investigate how add_axobserver can be used

"""

__all__ = ["is_in_ipython", "build_figlegend", "twin_axes_legend",
           "twin_fig_legend", "set_ytick", "set_xtick", "no_xtick_text",
           "no_ytick_text", "arrange_figures", "xlabel_fmt", "ylabel_fmt",
           "save_all_figures_to_pdf", "all_figures", "adjust_axwidth"]


def savefig(filename, *k, **kw):
    if kw.get("facetransparent", False):
        fig = pylab.gcf()
        alpha = fig.patch.get_alpha()
        fig.patch.set_alpha(0)
    pylab.savefig(filename, *k, **kw)
    if kw.get("facetransparent", False):
        fig.patch.set_alpha(alpha)


def xlabel_fmt(fmt, unit=None, axes=None):
    if axes is None:
        axes = plt.gca()
    if unit is None:
        axes.set_xlabel_fmt(fmt)
    else:
        axes.set_xlabel_fmt(fmt, unit)


def ylabel_fmt(fmt, unit=None, axes=None):
    if axes is None:
        axes = plt.gca()
    if unit is None:
        axes.set_ylabel_fmt(fmt)
    else:
        axes.set_ylabel_fmt(fmt, unit)

default_unit_names = {"s": "Time",
                      u"s": "Time",
                      "Hz": "Frequency",
                      u"Hz": "Frequency",
                      "m": "Length",
                      u"m": "Length",
                      }


class SimpleUnitFormatter(FormatStrFormatter):
    """Only adds the base unit does no prefix magic
    """
    def __init__(self, fmt, set_label_fun=None, label_fmt=None, unit=None):
        FormatStrFormatter.__init__(self, fmt)
        self.engfmt = EngFormatter(unit="", places=2)
        self.set_label_fun = set_label_fun
        self.default_unit = unit
        self._label_unit = None
        self.set_label_fmt(label_fmt)
        self.label_template = None

    def __call__(self, x, pos=None):
        div = 1
        prefix = ""
        for dig in range(3):
            digs = [abs((int((elem / div) * 10 ** dig) -
                        (elem / div) * 10 ** dig)) for elem in self.locs]
            if max(digs) < 0.001:
                self.fmt = "%%.%df" % dig
                break
        else:
            self.fmt = "%.3f"
        if self.set_label_fun and self.label_template:
            xlabel = self.label_template % dict(prefix=prefix)
            self.set_label_fun(xlabel)
        return FormatStrFormatter.__call__(self, x / div, pos)

    def get_label_fmt(self):
        return self._label_fmt

    def set_label_fmt(self, fmt):
        if fmt is None:
            self._label_fmt = ""
        else:
            self._label_fmt = fmt
        self.update_template()

    def get_label_unit(self):
        if self._label_unit is None:
            if self.default_unit is None:
                return ""
            else:
                return self.default_unit
        else:
            return self._label_unit

    def set_label_unit(self, unit):
        if unit is not None:
            self._label_unit = unit
        self.update_template()

    def update_template(self):
        fmt = self.get_label_fmt()
        if "[]" in fmt:
            if self.get_label_unit():
                fmt = fmt.replace("[]",
                                  "[%%(prefix)s%s]" % self.get_label_unit())
            else:
                fmt = fmt.replace("[]", "[%(prefix)sU]")
        elif "%(unit)s" in fmt:
            fmt = fmt % dict(unit=self.get_label_unit())
        self.label_template = fmt


class UnitFormatter(FormatStrFormatter):
    def __init__(self, fmt, set_label_fun=None,

                 label_fmt=None, unit=None, digs=3):
        FormatStrFormatter.__init__(self, fmt)
        self.engfmt = EngFormatter(unit="", places=2)
        self.set_label_fun = set_label_fun
        self.default_unit = unit
        self._label_unit = None
        self.set_label_fmt(label_fmt)
        self.label_template = None
        self.digs = digs

    def __call__(self, x, pos=None):
        locs = [abs(y) for y in self.locs if abs(y) != 0]
        if locs and max(locs) / min(locs) <= 10000:
            _, exponent = _help_format_sci(max(locs), 2)
            div = 10 ** exponent
            for dig in range(3):
                digs = [abs((int((elem / div) * 10 ** dig) -
                            (elem / div) * 10 ** dig)) for elem in self.locs]
                if max(digs) < 0.001:
                    self.fmt = "%%.%df" % dig
                    break
            else:
                self.fmt = "%%.%df" % self.digs
            if self.set_label_fun and self.label_template:
                prefix = si_exp_to_prefixes.get(exponent, "q")
                if prefix == "q":
                    prefix = ""
                    div = 1
                    self.fmt = "%%.%de" % self.digs
                xlabel = self.label_template % dict(prefix=prefix,
                                                    powerprefix=exponent)
                self.set_label_fun(xlabel)
            return FormatStrFormatter.__call__(self, x / div, pos)
        else:
            self.engfmt.locs = self.locs
            return self.engfmt(x, pos)

    def get_label_fmt(self):
        return self._label_fmt

    def set_label_fmt(self, fmt):
        if fmt is None:
            self._label_fmt = ""
        else:
            self._label_fmt = fmt
        self.update_template()

    def get_label_unit(self):
        if self._label_unit is None:
            if self.default_unit is None:
                return ""
            else:
                if self.default_unit == "deg":
                    return u"\xb0"
                else:
                    return self.default_unit
        else:
            if self._label_unit == "deg":
                return u"\xb0"
            else:
                return self._label_unit

    def set_label_unit(self, unit):
        if unit is not None:
            self._label_unit = unit
        self.update_template()

    def get_label_name_and_unit(self):
        unit = self.get_label_unit()
        default = "X"
        if unit == "Hz":
            default = "Frequency"
        elif unit in ["s", "h", "min"]:
            default = "Time"
        name = default_unit_names.get(unit, default)

        return dict(unit=unit, default=name)

    def update_template(self):
        fmt = self.get_label_fmt()
        if "[]" in fmt:
            if self.get_label_unit():
                fmt = fmt.replace("[]", u"[%(prefix)s{unit}]")
            else:
                fmt = fmt.replace("[]", u"[%(prefix)sU]")
        elif "[^]" in fmt:
            if self.get_label_unit():
                fmt = fmt.replace("[^]", ur"$[10^{{%%(powerprefix).0f" +
                                  ur"\rm{{{unit}}}}]$")
            else:
                fmt = fmt.replace("[^]", ur"$[10^{{%(powerprefix).0f}}]$")
        if "{unit}" in fmt or "{default}" in fmt:
            dct = self.get_label_name_and_unit()
            fmt = fmt.format(**dct)
        self.label_template = fmt


def get_info_names(x):
    return [y.name for y in x.dims]


class HFToolsAxes(Axes):
    name = "rectilinear"

    colorcycle = "bgrcmyk"
    markercycle = ['', 'o', 'v', '^', '<', '>', '*', 'x', '+']
    linecycle = ['-', '--', '-.', ':']

    def _plot_helper(self, x, y, *args, **kwargs):
        if not hasattr(y, "dims"):
            return Axes.plot(self, x, y, *args, **kwargs)
        if x.ndim == 1 and y.ndim == 1:
            return Axes.plot(self, x, y, *args, **kwargs)
        else:
            return Axes.plot(self, x, remove_tail(y), *args, **kwargs)

            Ns = y.shape[1:4]
            kw = kwargs.copy()
            lines = []
            if len(Ns) == 1:
                C = zip(itertools.cycle(self.colorcycle), range(Ns[0]))
                for c, i in C:
                    #kw.update(dict(color=c))
                    if hasattr(x, "dims") and (get_info_names(x) ==
                                               get_info_names(y)):
                        xx = x[:, i].squeeze()
                    else:
                        xx = x
                    lines.extend(Axes.plot(self, xx, remove_tail(y[:, i]),
                                           *args, **kw))
            elif len(Ns) == 2:
                C = zip(itertools.cycle(self.colorcycle), range(Ns[0]))
                M = zip(itertools.cycle(self.markercycle), range(Ns[1]))
                for c, i in C:
                    for m, j in M:
                        if hasattr(x, "dims") and (get_info_names(x) ==
                                                   get_info_names(y)):
                            xx = x[:, i, j].squeeze()
                        else:
                            xx = x
                        #kw.update(dict(color=c, marker=m))
                        lines.extend(Axes.plot(self, xx,
                                               remove_tail(y[:, i, j]),
                                               *args, **kw))
            elif len(Ns) > 2:
                C = zip(itertools.cycle(self.colorcycle), range(Ns[0]))
                M = zip(itertools.cycle(self.markercycle), range(Ns[1]))
                L = zip(itertools.cycle(self.linecycle), range(Ns[2]))
                for c, i in C:
                    for m, j in M:
                        for l, k in L:
                            if hasattr(x, "dims") and (get_info_names(x) ==
                                                       get_info_names(y)):
                                xx = x[:, i, j, k].squeeze()
                            else:
                                xx = x
                            #kw.update(dict(color=c, marker=m, line=l))
                            lines.extend(Axes.plot(self, xx,
                                                   remove_tail(y[:, i, j, k]),
                                                   *args, **kw))
            return lines

    def plot(self, *args, **kwargs):
        if "projection" in kwargs:
            projection = kwargs.pop("projection")
        else:
            projection = self.name
        vars = args[:2]
        args = args[2:]

        if len(vars) == 2 and isinstance(vars[1], (str, unicode)):
            args = (vars[1],) + args
            vars = vars[:1]

        if ((len(vars) == 1 and
             isinstance(vars[0], hfarray) and
             len(vars[0].dims) >= 1)):
            y = vars[0]
            x = hfarray(y.dims[0])
            vars = (x, y)

        if len(vars) == 1:
            y = vars[0]
            if projection in _projfun:
                x, y = _projfun[projection](None, y)
                return Axes.plot(self, y, *args, **kwargs)
            elif np.iscomplexobj(y):
                return Axes.plot(self, y.real, y.imag, *args, **kwargs)
            else:
                return Axes.plot(self, y, *args, **kwargs)
        elif len(vars) == 2:
            x = vars[0]
            y = vars[1]
            xunit = getattr(x, "unit", None)
            yunit = getattr(y, "unit", None)

            if projection in _projfun:
                x, y = _projfun[projection](x, y)
                lines = self._plot_helper(x, y, *args, **kwargs)
            elif np.iscomplexobj(y):
                xunit = yunit
                lines = self._plot_helper(y.real, y.imag, *args, **kwargs)
            else:
                lines = self._plot_helper(x, y, *args, **kwargs)
            if xunit:
                self.set_xlabel_unit(xunit)
            if yunit:
                self.set_ylabel_unit(yunit)
            return lines
        else:
            raise Exception("Missing plot data")

    def set_xlabel_unit(self, unit):
        xfmt = self.axes.xaxis.get_major_formatter()
        if hasattr(xfmt, "set_label_unit"):
            xfmt.set_label_unit(unit)

    def get_xlabel_unit(self):
        xfmt = self.axes.xaxis.get_major_formatter()
        if hasattr(xfmt, "get_label_unit"):
            return xfmt.get_label_unit()
        else:
            return None

    def set_ylabel_unit(self, unit):
        yfmt = self.axes.yaxis.get_major_formatter()
        if hasattr(yfmt, "set_label_unit"):
            yfmt.set_label_unit(unit)

    def set_xlabel_fmt(self, fmt, unit=None):
        xfmt = self.axes.xaxis.get_major_formatter()
        xfmt.set_label_fun = self.set_xlabel
        if hasattr(xfmt, "set_label_fmt"):
            self.set_xlabel_unit(unit)
            xfmt.set_label_fmt(fmt)

    def set_ylabel_fmt(self, fmt, unit=None):
        yfmt = self.axes.yaxis.get_major_formatter()
        yfmt.set_label_fun = self.set_ylabel
        if hasattr(yfmt, "set_label_fmt"):
            self.set_ylabel_unit(unit)
            yfmt.set_label_fmt(fmt)

    def get_ylabel_fmt(self):
        xfmt = self.axes.yaxis.get_major_formatter()
        if hasattr(xfmt, "get_label_fmt"):
            return xfmt.get_label_fmt()
        else:
            return None

    def get_xlabel_fmt(self):
        xfmt = self.axes.xaxis.get_major_formatter()
        if hasattr(xfmt, "get_label_fmt"):
            return xfmt.get_label_fmt()
        else:
            return None
matplotlib.projections.register_projection(HFToolsAxes)


class dBAxes(HFToolsAxes):
    name = "db"

    def __init__(self, *args, **kwargs):
        HFToolsAxes.__init__(self, *args, **kwargs)
        self.axes.xaxis.set_major_formatter(UnitFormatter("%.3f"))
        self.axes.yaxis.set_major_formatter(SimpleUnitFormatter("%.3f"))
        self.set_ylabel_fmt("[]", unit="dB")
        self.set_xlabel_fmt("{default} []")

    def set_xlabel_fmt(self, fmt, unit=None):
        HFToolsAxes.set_xlabel_fmt(self, fmt, unit)
matplotlib.projections.register_projection(dBAxes)


class dB10Axes(HFToolsAxes):
    name = "db10"

    def __init__(self, *args, **kwargs):
        HFToolsAxes.__init__(self, *args, **kwargs)
        self.axes.xaxis.set_major_formatter(UnitFormatter("%.3f"))
        self.axes.yaxis.set_major_formatter(SimpleUnitFormatter("%.3f"))
        self.set_ylabel_fmt("[]", unit="dB")
        self.set_xlabel_fmt("{default} []")

    def set_xlabel_fmt(self, fmt, unit=None):
        HFToolsAxes.set_xlabel_fmt(self, fmt, unit)
matplotlib.projections.register_projection(dB10Axes)


class MagAxes(HFToolsAxes):
    name = "mag"

    def __init__(self, *args, **kwargs):
        HFToolsAxes.__init__(self, *args, **kwargs)
        self.axes.xaxis.set_major_formatter(UnitFormatter("%.1f"))
        self.axes.yaxis.set_major_formatter(UnitFormatter("%.1f"))
        self.set_xlabel_fmt("{default} []")
        self.set_ylabel_fmt("[]")

    def set_xlabel_fmt(self, fmt, unit=None):
        HFToolsAxes.set_xlabel_fmt(self, fmt, unit)
matplotlib.projections.register_projection(MagAxes)


class MagSquareAxes(HFToolsAxes):
    name = "mag_square"

    def __init__(self, *args, **kwargs):
        HFToolsAxes.__init__(self, *args, **kwargs)
        self.axes.xaxis.set_major_formatter(UnitFormatter("%.1f"))
        self.axes.yaxis.set_major_formatter(UnitFormatter("%.1f"))
        self.set_xlabel_fmt("{default} []")
        self.set_ylabel_fmt("[]")

    def set_xlabel_fmt(self, fmt, unit=None):
        HFToolsAxes.set_xlabel_fmt(self, fmt, unit)
matplotlib.projections.register_projection(MagSquareAxes)


class UnityAxes(MagAxes):
    name = "unity"
matplotlib.projections.register_projection(UnityAxes)


class XSIAxes(MagAxes):
    name = "x-si"

    def __init__(self, *args, **kwargs):
        HFToolsAxes.__init__(self, *args, **kwargs)
        self.axes.xaxis.set_major_formatter(UnitFormatter("%.1f"))
        self.set_xlabel_fmt("{default} []")
matplotlib.projections.register_projection(XSIAxes)


class ComplexAxes(HFToolsAxes):
    name = "cplx"

    def __init__(self, *args, **kwargs):
        HFToolsAxes.__init__(self, *args, **kwargs)
        self.axes.xaxis.set_major_formatter(UnitFormatter("%.1f"))
        self.axes.yaxis.set_major_formatter(UnitFormatter("%.1f"))
        self.set_xlabel_fmt("Real []")
        self.set_ylabel_fmt("Imaginary []")
matplotlib.projections.register_projection(ComplexAxes)


class GroupDelayAxes(MagAxes):
    name = "groupdelay"

    def __init__(self, *args, **kwargs):
        MagAxes.__init__(self, *args, **kwargs)
        self.axes.yaxis.set_major_formatter(UnitFormatter("%.1f"))
        self.set_ylabel_fmt(r"$\tau$ []")
matplotlib.projections.register_projection(GroupDelayAxes)


class RealAxes(MagAxes):
    name = "real"
matplotlib.projections.register_projection(RealAxes)


class ImagAxes(MagAxes):
    name = "imag"
matplotlib.projections.register_projection(ImagAxes)


class DegAxes(MagAxes):
    name = "deg"

    def __init__(self, *args, **kwargs):
        MagAxes.__init__(self, *args, **kwargs)
        self.set_ylabel_fmt(u"[]")

    def set_ylabel_fmt(self, fmt, unit=u"\xb0"):
        MagAxes.set_ylabel_fmt(self, fmt, unit)
matplotlib.projections.register_projection(DegAxes)


class UnwrapDegAxes(DegAxes):
    name = "unwrapdeg"
matplotlib.projections.register_projection(UnwrapDegAxes)


class WrapUnwrapDegAxes(DegAxes):
    name = "wrapunwrapeddeg"
matplotlib.projections.register_projection(WrapUnwrapDegAxes)


class RadAxes(MagAxes):
    name = "rad"

    def __init__(self, *args, **kwargs):
        MagAxes.__init__(self, *args, **kwargs)
        self.set_ylabel_fmt(u"[]")

    def set_ylabel_fmt(self, fmt, unit=u"rad"):
        MagAxes.set_ylabel_fmt(self, fmt, unit)
matplotlib.projections.register_projection(RadAxes)


class UnwrapRadAxes(RadAxes):
    name = "unwraprad"
matplotlib.projections.register_projection(UnwrapRadAxes)


class ComplexPolarAxes(PolarAxes):
    name = "cplxpolar"

    def plot(self, *args, **kwargs):
        projection = self.name
        vars = args[:2]
        args = args[2:]
        if len(vars) == 2 and isinstance(vars[1], (str, unicode)):
            args = (vars[1],) + args
            vars = vars[:1]

        if ((len(vars) == 1 and
             isinstance(vars[0], hfarray) and
             len(vars[0].dims) >= 1)):
            y = vars[0]
            x = hfarray(y.dims[0])
            vars = (x, y)

        if len(vars) == 1:
            y = vars[0]
            if projection in _projfun:
                x, y = _projfun[projection](None, y)
                return Axes.plot(self, y, *args, **kwargs)
            elif np.iscomplexobj(y):
                return Axes.plot(self, y.real, y.imag, *args, **kwargs)
            else:
                return Axes.plot(self, y, *args, **kwargs)
        elif len(vars) == 2:
            x = vars[0]
            y = remove_tail(vars[1])

            if projection in _projfun:
                x, y = _projfun[projection](x, y)
                lines = Axes.plot(self, x, y, *args, **kwargs)
            elif np.iscomplexobj(y):
                lines = Axes.plot(self, y.real, y.imag, *args, **kwargs)
            else:
                lines = Axes.plot(self, x, y, *args, **kwargs)
#            if xunit:
#                self.set_xlabel_unit(xunit)
#            if yunit:
#                self.set_ylabel_unit(yunit)
            return lines
        else:
            raise Exception("Missing plot data")
matplotlib.projections.register_projection(ComplexPolarAxes)


def cplx_polar_projection(x, y):
    if x is None:
        r = abs(y)
        theta = np.angle(y)
    elif np.iscomplexobj(y):
        r = abs(y)
        theta = np.angle(y)
    else:
        y = x + 1j * y
        r = abs(y)
        theta = np.angle(y)
    return theta, r

_projfun = dict(db=lambda x, y: (x, hftools.math.dB(y)),
                db10=lambda x, y: (x, 10 * np.log10(abs(y))),
                mag=lambda x, y: (x, hfarray(np.abs(y), dims=y.dims)),
                mag_square=lambda x, y: (x, hfarray(np.abs(y) ** 2,
                                                    dims=y.dims)),
                rad=lambda x, y: (x, hfarray(np.angle(y, deg=False),
                                             dims=y.dims)),
                deg=lambda x, y: (x, hfarray(np.angle(y, deg=True),
                                             dims=y.dims)),
                unwraprad=lambda x, y: (x, unwrap_phase(y)),
                unwrapdeg=lambda x, y: (x, unwrap_phase(y, deg=True)),
                imag=lambda x, y: (x, hfarray(np.imag(y), dims=y.dims)),
                real=lambda x, y: (x, hfarray(np.real(y), dims=y.dims)),
                groupdelay=lambda x, y: delay(x, y),
                cplxpolar=cplx_polar_projection)
_projfun["x-si"] = lambda x, y: (x, y)
_projfun[None] = lambda x, y: (x, y)


def is_in_ipython():
    """Funktionen returnerar True om scriptet kors under ipython

    """
    return "IPython" in sys.modules


def all_figures():
    """Return all open figures
    """
    return [fig.canvas.figure for _, fig in sorted(Gcf.figs.items())]


def adjust_axwidth(dw, axes=None):
    """Change width of axes.

        *dw* is added to axes width
        *axes* None | axes | [axes,...], if None all axes of current figure
    """
    if axes is None:
        axes = plt.gcf().axes
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax in axes:
        bbox = ax.get_position()
        bbox.x1 += dw
        ax.set_position(bbox)


def save_all_figures_to_pdf(filename, *kw):
    filename = hftools.path(filename)
    if filename.splitext()[1].lower() != ".pdf":
        raise ValueError("save_all_figures_to_pdf can only handle pdf files.")
    pdffile = PdfPages(filename)
    for fig in all_figures():
        fig.savefig(pdffile, format="pdf")
    pdffile.close()


def build_figlegend(axes=None):
    """Ta fram listor med linjer och etiketter fran *axes* lista.
    Om *axes* ar None sa anvand lista med axes fran aktiv figur.
    Anvands for att generera *lines* och *labels* till
    pylab.figlegend-anrop. Ignorerar labels som borjar med _

    Example::

        (lines,labels)=build_figlegend()
        pylab.figlegend(lines, labels, loc="best")

    """
    if axes is None:
        axes = pylab.gcf().axes
    labels = []
    lines = []
    for ax in axes:
        for line in ax.lines:
            if not line.get_label().startswith("_"):
                labels.append(line.get_label())
                lines.append(line)
    return lines, labels


def twin_axes_legend(axes=None, loc="best", **kw):
    """Skapa axes legend for alla linjer i en plot som skapats med
    dubbla y-axlar (twinx)
    """
    (lines, labels) = build_figlegend(axes)
    pylab.legend(lines, labels, loc=loc, **kw)


def twin_fig_legend(axes=None, loc="best", **kw):
    """Skapa figur legend for alla linjer i en plot som skapats med
    dubbla y-axlar (twinx)
    """
    (lines, labels) = build_figlegend(axes)
    pylab.figlegend(lines, labels, loc=loc, **kw)


def set_ytick(major, minor, axes=None):
    """Satt *major* och *minor* tick-marks pa y-axeln
    """
    if axes is None:
        axes = pylab.gca()
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax in axes:
        ax.yaxis.set_major_locator(pylab.MultipleLocator(major))
        ax.yaxis.set_minor_locator(pylab.MultipleLocator(minor))


def set_xtick(major, minor, axes=None):
    """Satt *major* och *minor* tick-marks pa x-axeln
    """
    if axes is None:
        axes = pylab.gca()
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax in axes:
        ax.xaxis.set_major_locator(pylab.MultipleLocator(major))
        ax.xaxis.set_minor_locator(pylab.MultipleLocator(minor))


def no_xtick_text():
    """Ta bort text kopplad till y-ticks
    """
    ticks, _ = pylab.xticks()
    ticks, _ = pylab.xticks(ticks, [""] * len(ticks))


def no_ytick_text():
    """Ta bort text kopplad till y-ticks
    """
    ticks, _ = pylab.yticks()
    ticks, _ = pylab.yticks(ticks, [""] * len(ticks))


def fig_fits_w(fig, x):
    """Lista ut of figuren *fig* far plats pa bredden pa skarmen vid
    position *x*, *y*
    """
    w, _ = _get_max_width()
    win_w = fig.window.winfo_width()
    result = (x + win_w) < w
    return result


def fig_fits_h(fig, y):
    """Lista ut of figuren *fig* far plats pa hojden pa skarmen vid
    position *x*, *y*
    """
    _, h = _get_max_width()
    win_h = fig.window.winfo_height()
    result = (y + win_h) < h
    return result

DELTAOFFSET = 10


def arrange_figures(layout=None, screen=2, xgap=10,
                    ygap=30, offset=0, figlist=None):
    """Automatiskt arrangera alla figurer i ett icke overlappande
    monster

       *layout*
            Anvands inte just nu

       *screen* [=2]
            anger vilken skarm man i forsta hand vill ha fonstren pa.

       *xgap*
            Gap i x-led mellan fonster

       *ygap*
            Gap i y-led mellan fonster

       *offset*
            Nar skarmen ar fylld borjar man om fran ovre hogra hornet
            men med en offset i x och y led.

       *figlist*
            Lista med figurnummer som skall arrangeras

    """
    #Hamta information om total skarmbredd over alla anslutna skarmar
    if not is_in_ipython():
        return
#    pylab.show()
    pylab.ioff()
    x0 = 0 + offset
    y0 = 0 + offset
    if screen == 2:
        x0 = (pylab.get_current_fig_manager().window.winfo_screenwidth() +
              offset)
    if figlist is None:
        figlist = sorted([x for x in Gcf.figs.iteritems()])

    x = x0
    y = y0
    maxheight = 0
    while figlist:
        fig = _, f = figlist[0]
        figlist = figlist[1:]
        if fig_fits_w(f, x):
            move_fig(f, x, y)
            x = x + f.window.winfo_width() + xgap
            maxheight = max(maxheight, f.window.winfo_height())
        else:
            x = x0
            y = y + maxheight + ygap
            maxheight = 0
            if fig_fits_h(f, y):
                move_fig(f, x, y)
                x = x + f.window.winfo_width() + xgap
            else:
                arrange_figures(offset=DELTAOFFSET, xgap=xgap, ygap=ygap,
                                screen=screen, figlist=[fig] + figlist)
                break
    pylab.ion()


def _get_max_width():
    """Hamta information om total skarmbredd och -hojd
    """
    from win32api import GetSystemMetrics
    #Hamta information om total skarmbredd over alla anslutna skarmar
    width = GetSystemMetrics(78)
    #Hamta information om total skarmhojd over alla anslutna skarmar
    height = GetSystemMetrics(79)
    return width, height


def move_fig(fig, x, y):
    """Flytta figur *fig*, till *x*  och *y* koordinaten men
    tillat inte att det hamnar utanfor skarmkanten.

    """
    pylab.ioff()
    w, h = _get_max_width()
    win_w = fig.window.winfo_width()
    win_h = fig.window.winfo_height()

    #Flytta fonster sa det inte hamnar utanfor skarmen
    x0 = max(0, min(x, w - win_w))
    y0 = max(0, min(y, h - win_h))
    fig.window.wm_geometry("+%d+%d" % (x0, y0))
    pylab.ion()


def move_current_figure(x, y):
    """Flytta aktiv figur, till *x*  och *y* koordinaten men
    tillat inte att det hamnar utanfor skarmkanten.

    """

    move_fig(pylab.get_current_fig_manager(), x, y)


def set_figure_positions(lista):
    """Flytta ett antal figurer till angivna positioner

         *lista*
            [(fig1, (x1, y1)), (fig2, (x2, y2)), ...]

    """
    for (i, (x, y)) in lista:
        move_fig(Gcf.figs[i], x, y)

_reg_split_wm_geometry = re.compile("[x+]")


def get_current_figure_position(fig=None):
    """Hamta position for figur *fig*

        *fig*
            Figur nummer, None anger aktiv figur.
    """
    if fig is None:
        fig = pylab.get_current_fig_manager()
    else:
        fig = Gcf.figs[fig]
    wm_geom = fig.window.wm_geometry()
    (x, y) = map(int, _reg_split_wm_geometry.split(wm_geom))[2:]
    return x, y


def get_figure_positions():
    """Hamta position for alla figurer

    """
    return [(i, get_current_figure_position(i)) for i in Gcf.figs.keys()]


if __name__ == '__main__':
    from hftools import path
    from hftools.file_formats import read_data
    datapath = (path(hftools.__file__).dirname() / "../data/").normpath()

    fwd = read_data(datapath / "fwd.s2p")
    cold = read_data(datapath / "cold.s2p")
    active = read_data(datapath / "active.s2p")

    plt.clf()
    plt.subplot(111, projection="db")
    plt.plot(fwd.S[..., 0, 0])
