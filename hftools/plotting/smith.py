# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import sys, os, pdb, itertools
import numpy as np

from numpy import array, zeros, ones, newaxis, sqrt, pi, exp, sin, cos, tan, \
    log, log10, arange, zeros_like, empty_like, linspace, angle

import pylab
import matplotlib
from pylab import clf, plot, legend, axis, xlabel, ylabel, title, subplot, \
    savefig, semilogx, semilogy, errorbar, hist, figtext, figure, axhline,\
    suptitle, twinx, grid, gca, Circle, Line2D, Polygon
from matplotlib.patches import Arc
import matplotlib.spines as mspines

from hftools.plotting.helper import HFToolsAxes

class SmithAxes(HFToolsAxes):
    name = "smith"
    def __init__(self, *args, **kwargs):
        HFToolsAxes.__init__(self, *args, **kwargs)
        smith(self)

#    def _gen_axes_patch(self):
#        return Circle((0.5, 0.5), 0.5)

matplotlib.projections.register_projection(SmithAxes)

class ComplexPolarAxes(HFToolsAxes):
    name = "cpolar"
    def __init__(self, *args, **kwargs):
        HFToolsAxes.__init__(self, *args, **kwargs)
        polar_grid(self)

#    def _gen_axes_patch(self):
#        return Circle((0.5, 0.5), 0.5)

matplotlib.projections.register_projection(ComplexPolarAxes)

__all__ = ["smith", "inv_smith", "smith_polar"]
def angle(x, deg=False, branch=0):
    if deg:
        add = 360
        offset = exp(1j * branch / 180. * pi)
    else:
        add = 2*pi
        offset = exp(1j * branch)
    x = x/offset
    a = np.angle(x, deg)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        a[a < 0] += add
    else:
        if a < 0:
            a += add
    a += branch
    return a

class MyCircle(object):
    def __init__(self, x, y, r, t1=None, t2=None):
        self.x = x
        self.y = y
        self.r = r
        self.t1 = t1
        self.t2 = t2
        self.flipx = 1
        self.flipy = 1

    @property
    def pars(self):
        return (self.x, self.y), self.r

    def __repr__(self):
        return "%s%r"%(self.__class__.__name__, (self.x, self.y, self.r, self.t1, self.t2))

    def get_artist(self, *k, **kw):
        kw = kw.copy()
        if "fc" not in kw:
            kw["fc"] = "none"
        if self.t1 is None or self.t2 is None:
            return Circle((self.x, self.y), self.r, *k, **kw)
        else:
            pars = (self.x, self.y), self.flipx*2*self.r,  self.flipy*2*self.r, 0, self.t1, self.t2
            return Arc(*(pars+k), **kw)


class MyLine(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @property
    def xdata(self):
        return [self.x1, self.x2]

    @property
    def ydata(self):
        return [self.y1, self.y2]

    @property
    def pars(self):
        return self.xdata, self.ydata

    def __repr__(self):
        return "%s%r"%(self.__class__.__name__, (self.x1, self.y1, self.x2, self.y2))

    def get_artist(self, *k, **kw):
        kw = kw.copy()
        if "fc" in kw:
            del kw["fc"]
        return Polygon(array([(self.x1, self.y1), (self.x2,self.y2)]), *k, **kw)

def z2gamma(z):
    if np.isinf(abs(z)):
        if np.imag(z) >= 0:
            return 1+0j
        else:
            return 1-1e-15j
    else:
        return (z - 1) / (z + 1)

def y2gamma(y):
    if np.isinf(abs(y)):
        if np.imag(y) <= 0:
            return -1+1e-15j
        else:
            return -1-1e-15j
    else:
        return (1 - y) / (1 + y)

def RCircle(r, x1=None, x2=None):
    x0, y0 = (r/(r + 1.), 0)
    radius =  abs(1/(r + 1.))
    t1 = t2 = None
    if x1 is None or x2 is None:
        pass
    else:
        x1, x2 = min(x1, x2), max(x1,x2)
        z1 = complex(r, x1)
        z2 = complex(r, x2)
        c1 = z2gamma(z1)
        c2 = z2gamma(z2)
        c0 = complex(x0, y0)
        t1 = angle(c2-c0, deg=True)
        t2 = angle(c1-c0, deg=True)
    return MyCircle(x0, y0, radius, t1, t2)

def XCircle(x, r1=None, r2=None):
    if x==0:
        if r1 is None or r2 is None:
            return MyLine(-1, 0, 1, 0)
        else:
            x1 = (z2gamma(r1)+0j).real
            x2 = (z2gamma(r2)+0j).real
            return MyLine(x1, 0, x2, 0)

    x = float(x)
    x0, y0 = (1, 1/x)
    radius = abs(1/x)
    t1 = t2 = None
    if r1 is None or r2 is None:
        pass
    else:
        r1, r2 = sorted((r1, r2))
        z1 = complex(r1, x)
        z2 = complex(r2, x)
        c1 = z2gamma(z1)
        c2 = z2gamma(z2)
        c0 = complex(x0, y0)
        t1, t2 = sorted((angle(c2-c0, deg=True), angle(c1-c0, deg=True)))
    return MyCircle(x0, y0, radius, t1, t2)


def GCircle(g, b1=None, b2=None):
    x0, y0 = (-g/(g + 1.), 0)
    radius =  abs(1/(g + 1.))
    t1 = t2 = None
    if b1 is None or b2 is None:
        pass
    else:
        y1 = complex(g, b2)
        y2 = complex(g, b1)
        c1 = y2gamma(y1)
        c2 = y2gamma(y2)
        c0 = complex(x0, y0)
        t2 = angle(c2-c0, deg=True, branch=180)
        t1 = angle(c1-c0, deg=True, branch=180)
    return MyCircle(x0, y0, radius, t1, t2)


def BCircle(b, g1=None, g2=None):
    if b==0:
        if g1 is None or g2 is None:
            return MyLine(-1, 0, 1, 0)
        else:
            x1 = (z2gamma(g1)+0j).real
            x2 = (z2gamma(g2)+0j).real
            return MyLine(x1, 0, x2, 0)
    if b >= 0:
        branch = 90
        g1, g2 = (g2, g1)
    else:
        branch = -90
    b = float(b)
    x0, y0 = (-1, -1/b)
    radius = abs(1/b)
    t1 = t2 = None
    if g1 is None or g2 is None:
        pass
    else:
        g1, g2 = (g1, g2)
        z1 = complex(g1, b)
        z2 = complex(g2, b)
        c1 = y2gamma(z1)
        c2 = y2gamma(z2)
        c0 = complex(x0, y0)
        t1, t2 = (angle(c2-c0, deg=True, branch=branch), angle(c1-c0, deg=True, branch=branch))
    return MyCircle(x0, y0, radius, t1, t2)

def polar_grid(ax=None, mode="full", **kw):
    if ax is None:
        ax = gca()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_frame_on(False)
    ax.axis("equal")
    ax.axis([-1.01,1.01,-1.01,1.01])
    ax.add_patch(Circle((0, 0), 1, fc="w", ec="k"))
    angles = dict(full=(0, 360), lower=(180, 360.00001), upper=(0, 180.000001))
    t1, t2 = angles[mode]
    for r in arange(0, 1.001, 0.2):
        ax.add_patch(Arc((0,0), 2*r, 2*r, 0, t1, t2, color="k"))

    for fi in np.arange(t1/180*pi, t2/180*pi, pi/4):
        z = exp(1j*fi)
        ax.add_patch(Polygon(array([[0, 0], [z.real, z.imag]]), color="k"))


def smith_grid(ax=None, mode="full", standard=True, **kw):
    if ax is None:
        ax = gca()

    if standard:
        angles = dict(full=(0, 360), lower=(180, 360), upper=(0, 180))
        xlims = dict(full=(-np.inf, np.inf),
                     lower=(-np.inf, 0),
                     upper=(0, np.inf))
        RC = RCircle
        XC = XCircle
        upper = [1./3, 1, 3]
        lower = [-1./3, -1, -3]

    else:
        angles = dict(full=(0, 360), lower=(0, 180), upper=(180, 360))
        xlims = dict(full=(-np.inf, np.inf),
                     upper=(-np.inf, 0),
                     lower=(0, np.inf))
        lower = [1./3, 1, 3]
        upper = [-1./3, -1, -3]
        RC = GCircle
        XC = BCircle
    t1, t2 = angles[mode]
    x1, x2 = xlims[mode]
    ax.add_patch(Arc((0,0), 2, 2, 0, t1, t2, fc="w", ec="k"))

    for r in [0, 1./3, 1, 3]:
        c = RC(r, x1, x2)
        ax.add_patch(c.get_artist(color="k"))

    xcircles = [0]
    if mode in ["full", "lower"]:
        xcircles.extend(lower)
    if mode in ["full", "upper"]:
        xcircles.extend(upper)

    for x in xcircles:
        circ = XC(x, 0, np.inf)
        ax.add_patch(circ.get_artist(color="k"))

def smith(ax=None):
    if ax is None:
        ax = gca()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_frame_on(False)
    ax.axis("equal")
    ax.axis([-1.01,1.01,-1.01,1.01])
    ax.add_patch(Circle((0, 0), 1, fc="w", ec="k"))
    smith_grid(ax)

def inv_smith(ax=None, **kw):
    ax = empty_grid(ax)
    ax.add_patch(Circle((0, 0), 1, fc="w", ec="k"))
    smith_grid(ax, standard=False)
    return

def smith_lower(ax=None):
    if ax is None:
        ax = gca()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_frame_on(False)
    ax.axis("equal")
    ax.axis([-1.01,1.01,-1.01,1.01])

    smith_grid(ax, "lower")

def smith_upper(ax=None):
    if ax is None:
        ax = gca()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_frame_on(False)
    ax.axis("equal")
    ax.axis([-1.01,1.01,-1.01,1.01])
    smith_grid(ax, "upper")


def smith_polar(ax=None):
    if ax is None:
        ax = gca()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_frame_on(False)
    ax.axis("equal")
    ax.axis([-1.01,1.01,-1.01,1.01])

    ax.add_patch(Circle((0, 0), 1, fc="w", ec="k"))
    smith_lower(ax)
    polar_grid(ax, "upper")

def empty_grid(ax=None):
    if ax is None:
        ax = gca()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_frame_on(False)
    ax.axis("equal")
    ax.axis([-1.01, 1.01, -1.01, 1.01])
    return ax

if __name__ == "__main__":
    clf()
    ax1 = subplot(331)
    smith(ax1)
    ax2 = subplot(332)
    inv_smith(ax2)
    ax3 = subplot(333)
    smith_polar(ax3)

    ax4 = subplot(334)
    empty_grid(ax4)
    smith_grid(ax4)
    smith_grid(ax4, standard=False)

    ax5 = subplot(335)
    empty_grid(ax5)
    polar_grid(ax5)

    ax6 = subplot(336)
    empty_grid(ax6)
    polar_grid(ax6, "lower")

    ax7 = subplot(337)
    empty_grid(ax7)
    polar_grid(ax7, "upper")
    smith_grid(ax7, "lower", standard=False)

