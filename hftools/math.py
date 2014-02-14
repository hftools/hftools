# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import numpy as np
import numpy.linalg as linalg

from numpy import pi, exp, array, zeros, sqrt
from numpy.lib.stride_tricks import broadcast_arrays, as_strided

from hftools.dataset import make_same_info_list, hfarray,\
    DimMatrix_i, DimMatrix_j
from hftools.dataset.arrayobj import _hfarray, make_same_info


def angle(z, deg=False):
    res = np.angle(z, deg)
    if isinstance(z, _hfarray):
        return z.__class__(res, dims=z.dims, copy=False)
    return res


def make_matrix(a, b, c, d):
    a, b, c, d = make_same_info_list([a, b, c, d])
    abcdshape = zip(a.shape, b.shape, c.shape, d.shape)
    maxshape = (tuple(max(x) for x in abcdshape) + (2, 2))
    res = zeros(maxshape, a.dtype)
    res[..., 0, 0] = a
    res[..., 0, 1] = b
    res[..., 1, 0] = c
    res[..., 1, 1] = d
    dims = a.dims + (DimMatrix_i("i", 2), DimMatrix_j("j", 2),)
    out = hfarray(res, dims=dims)
    return out


def chop(x, threshold=1e-16):
    """Round numbers with magnitude smaller than *threshold* to zero
    """
    if isinstance(x, np.ndarray):
        x = x.copy()
        x[abs(x) < threshold] = 0
        return x
    else:
        if abs(x) < threshold:
            return 0 * x
        else:
            return x


def dB(x):
    """Convert x from dB to linear (voltage).

    ..math::    dB(x) = 20 ln(|x|)
    """
    return 20 * np.log10(abs(x))


def dBinv(x):
    """Convert x from dB to linear (voltage).

    ..math::    dBinv(x) = 10^{x/20}
    """
    return 10**(x / 20.)

#
# Convert to complexform
#


def dB_angle_to_complex(mag, ang):
    """Convert magnitude and angle to complex value

    *mag*   magnitude in dB
    *ang*   angle in degrees
    """
    return dBinv(mag) * exp(1j * pi * ang / 180.)


def mag_angle_to_complex(mag, ang):
    """Convert magnitude and angle to complex value

    *mag*   magnitude in linear scale
    *ang*   angle in degrees
    """
    return mag * exp(1j * pi * ang / 180.)


def re_im_to_complex(realpart, imaginarypart):
    """Convert real and imaginary parts to complex value

    *realpart*        real part linear units
    *imaginarypart*   imaginary part linear units
    """
    return realpart + 1j*imaginarypart


def continous_phase_sqrt(q):
    phase = unwrap_phase(q)
    Q = sqrt(abs(q)) * exp(1j * phase / 2)
    return Q


def _unwrap_phase(data, deg=False):
    """Virar upp fasen, dvs forsoker ta bort eventuella fashopp.

       >>> x = np.arange(0, 11, 1.5)
       >>> y = np.exp(1j*x)
       >>> np.angle(y)
       array([ 0.        ,  1.5       ,  3.        , -1.78318531, -0.28318531,
               1.21681469,  2.71681469, -2.06637061])
       >>> unwrap_phase(hfarray(y))
       hfarray([  0. ,   1.5,   3. ,   4.5,   6. ,   7.5,   9. ,  10.5])
    """
    result = data.real * 0
    a = angle(data[0])
    a0 = np.median(array(a.flat[0], copy=False))
    add_pos = abs(a - a0 + 2 * pi) < 2
    add_neg = abs(a - a0 - 2 * pi) < 2
    result[0] = a
    result[1:] = (np.add.accumulate(angle(data[1:] / data[:-1]), 0) +
                  angle(data[0]))
    result1 = np.where(array(add_pos), array(result + 2 * pi), array(result))
    result2 = np.where(array(add_neg), array(result1 - 2 * pi), array(result1))
    if deg:
        result2 = result2 / pi * 180
    return hfarray(result2, dims=data.dims)


def unwrap_phase(data, deg=False):
    a1 = _unwrap_phase(data, deg=deg)
    return a1


def delay(freq, var):
    r"""Berakna grupploptid :math:`\tau=-\frac{d\varphi}{d\omega}` for data

    Invariabler

        *freq*
            frekvenser som *var* galler for

        *var*
            komplexa data som skall anvandas i berakningen

    Resultat (f,tau)

        *f*
            frekvenserna som motsvarar mittpunkter i *freq*

        *tau*
            beraknad grupploptid med hjalp av mittpunkts approximation for
            derivatan

   >>> x=hfarray(np.arange(0,10.))
   >>> y=hfarray(np.exp(-2j*np.pi*x/10))
   >>> f,tau=delay(x,y)
   >>> f
   hfarray([ 0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5])
   >>> tau
   hfarray([ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1])

    """
    freq, var = make_same_info(freq, var)
    f_mean = (freq[:-1] + freq[1:]) / 2
    domega = (2 * pi * (freq[:-1] - freq[1:]))
    dfi = angle(var[1:] / var[:-1])
    return (f_mean, dfi / domega)


def smooth(data, aperture, axis=0):
    """Smoothar *data* med aritmetiskt medelvarde langs med *axis* [=0].

    Invariabler

        *data*
            data som skall smoothas

        *aperture*
            hur manga sampel som skall medlvardesbildas for varje punkt
            i smoothingen

        *axis*
            axel som skall smoothas langs default = 1
    """
    data = np.asanyarray(data)
    newdata = np.empty_like(data)
    len = data.shape[axis] - 1
    if aperture % 2 == 0:
        odd = 0
    else:
        odd = 1
    for i in range(data.shape[axis]):
        wid = min(i, aperture // 2, abs(len - i))
        if wid != 0:
            newdata[i] = np.mean(data[i - wid: i + wid + odd], axis)
        else:
            newdata[i] = data[i]
    return newdata


def poly_smooth(x, y, aperture, axis=0, N=3):
    """Smoothar *data* med hjalp av *N* te gradens polynom, langs med
    *axis* [=0].

    Invariabler

        *x*
            x-varden for data som skall smoothas

        *y*
            y-varden for data som skall smoothas

        *aperture*
            hur manga sampel som skall medlvardesbildas for varje punkt i
            smoothingen

        *axis*
            axel som skall smoothas langs

        *N*
            Vilken grad skallpolynomet ha

    """
    import scipy
    newdata = np.empty_like(y)
    size = x.shape[axis] - 1
    if aperture % 2 == 0:
        odd = 0
    else:
        odd = 1
    wid = aperture // 2

    for i in range(x.shape[axis]):
        start = min(max(i - wid, 0), max(0, size - aperture))
        stop = max(min(i + wid + odd, size), aperture)
        assert stop - start == aperture
        poly = scipy.polyfit(x[start:stop], y[start:stop], N)
        newdata[i] = scipy.polyval(poly, x[i])
    return newdata


def poly_smooth_magphase(x, data, aperture, axis=0, N=3):
    """Smoothar magnitud och vinkel hos *data* med hjalp av
    :func:`poly_smooth`.

    """
    m = poly_smooth(x, abs(data), aperture, axis, N)
    p = poly_smooth(x, unwrap_phase(data), aperture, axis, N)
    return m * exp(1j * p)


def smooth_magphase(data, aperture, axis=0):
    """Smoothar magnitud och vinkel hos *data* med hjalp av
    :func:`smooth`.

    """
    m = smooth(abs(data), aperture, axis)
    p = smooth(unwrap_phase(data), aperture, axis)
    return m * exp(1j * p)


def linear_extrapolate(x, y, xi):
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    xi = np.asanyarray(xi)
    idx0 = np.searchsorted(x[1:-1], xi)
    idx1 = idx0 + 1
    x0 = x[idx0]
    x1 = x[idx1]
    y0 = y[idx0]
    y1 = y[idx1]
    d0 = (xi - x0)
    d1 = (x1 - xi)
    d = (x1 - x0)
    return (d0 * y1 + d1 * y0) / d


def interpolate(x, y, xi):
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    xi = np.asanyarray(xi)
    if xi.min() < x.min():
        raise ValueError("Can not interpolate below lowest value")
    elif xi.max() > x.max():
        raise ValueError("Can not interpolate below highest value")
    else:
        return linear_extrapolate(x, y, xi)


#def check_is_multi_matrix(x):
#    """A multimatrix is a matrix on the last N indices"""

def firstpos(x, N=2):
    return as_strided(x, x.shape[:-N], x.strides[:-N])


def firstelement(x, N=2):
    return as_strided(x, x.shape[-N:], x.strides[-N:])


def get_shape_helper(x, N):
    if N == 0:
        return tuple()
    else:
        return x[-N:]


def get_info_helper(x, N):
    if N == 0:
        return tuple()
    else:
        return x[:-N]


class broadcast_matrices(object):
    def __init__(self, a, N=2):
        arrays = a
        if all(isinstance(x, _hfarray) for x in a):
            arrays = make_same_info_list(arrays)
        else:
            raise Exception("Can only broadcast hfarrays")
        try:
            N[0]
        except TypeError:
            N = [N] * len(a)
        self.Nlist = Nlist = N

        _matrixshapes = [get_shape_helper(x.shape, Nelem)
                                              for x, Nelem in zip(arrays, Nlist)]
        _matrixstrides = [get_shape_helper(x.strides, Nelem)
                                                for x, Nelem in zip(arrays, Nlist)]
        self._matrixshapes = _matrixshapes
        self._matrixstrides = _matrixstrides
        infos = [x.info for x, Nelem in zip(arrays, Nlist)]

        firstelems = broadcast_arrays(*[firstpos(x, Nelem)
                                        for x, Nelem in zip(arrays, Nlist)])
        self._broadcasted = broadcasted = []
        for o, endshapes, endstrides, info in zip(firstelems, _matrixshapes,
                                                  _matrixstrides, infos):
            x = as_strided(o, o.shape + endshapes, o.strides + endstrides)
            broadcasted.append(hfarray(x, dims=info, copy=False))

        self.outershape = broadcasted[0].shape[:-Nlist[0]]

    def __iter__(self):
        broadcasted = self._broadcasted
        to_enumerate = firstpos(broadcasted[0], self.Nlist[0])
        for index, _ in np.ndenumerate(to_enumerate):
            yield [x.__getitem__(index) for x in broadcasted]


def inv(A):
    inv = linalg.inv
    result = inv(A)
    result = hfarray(result, dims=A.info)
    return result


def matrix_multiply_old(A, B):
    dot = np.dot
    res = broadcast_matrices((A, B))
    x = dot(firstelement(A), firstelement(B))
    resinfo = res._broadcasted[0].info
    resempty = np.empty(res.outershape + x.shape, dtype=x.dtype)
    result = hfarray(resempty, dims=resinfo)
    for a, b, r in broadcast_matrices((A, B, result)):
        r[...] = dot(a, b)
    return result


def matrix_multiply(a, b):
    """Multiply arrays of matrices.
    
    a and b are hfarrays containing dimensions DimMatrix_i and DimMatrix_j.
    Matrix multiplication is done by broadcasting the other dimensions first.
    """
    A, B = make_same_info(a, b)
    res = np.einsum("...ij,...jk->...ik", A, B)
    return hfarray(res, dims=A.info)


def flatten_non_matrix(A):
    out = np.ascontiguousarray(A)
    shape = (int(np.multiply.reduce(A.shape[:-2])),) + A.shape[-2:]
    out.shape = shape
    return out


def det(A):
    det = linalg.det
    result = det(A)
    result = hfarray(result, dims=A.info[:-2])
    return result


def solve_Ab(A, b, squeeze=True):
    AA, bb = make_same_info(A, b)
    x = np.linalg.solve(AA, bb)
    result = hfarray(x, dims=bb.info)
    if squeeze:
        result = result.squeeze()
    return result


def lstsq(A, b, squeeze=True):
    lstsq = np.linalg.lstsq
    res = broadcast_matrices((A, b))
    x, residuals, rank, s = lstsq(firstelement(res._broadcasted[0]), firstelement(res._broadcasted[1]))
    xinfo = res._broadcasted[0].info
    xempty = np.empty(res.outershape + x.shape, dtype=x.dtype)
    xresult = hfarray(xempty, dims=xinfo)

    for a, b, rx in broadcast_matrices((A, b, xresult)):
        rx[...], _, _, _ = lstsq(a, b)
    return xresult


if __name__ == '__main__':
    a = array([[[1., 2], [3, 4]]])
    da = array([0.])
    b = array([[[4., 3], [2, 1]],
               [[5, 3], [2, 1]],
               [[7, 3], [2, 1]]])
    db = array([0, 0, 0.])
    m = broadcast_matrices((a, b))
    m2 = broadcast_matrices((b, db), (2, 0))
