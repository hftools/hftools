#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import os

import numpy as np

import hftools.math as hfmath
from hftools.dataset import hfarray, DimSweep, DimMatrix_i, DimMatrix_j
from hftools.testing import TestCase, make_load_tests

basepath = os.path.split(__file__)[0]
load_tests = make_load_tests(hfmath)


class Test_angle(TestCase):
    def setUp(self):
        self.A = hfarray([1, 1j, -1, -1j])
        self.a = np.array([1, 1j, -1, -1j])

    def test_va_1(self):
        res = hfmath.angle(self.A, deg=False)
        self.assertAllclose(res, np.array([0, np.pi / 2, np.pi, -np.pi / 2]))

    def test_va_2(self):
        res = hfmath.angle(self.A, deg=True)
        self.assertAllclose(res, np.array([0, 90, 180, -90]))

    def test_np_1(self):
        res = hfmath.angle(self.a, deg=False)
        self.assertAllclose(res, np.array([0, np.pi / 2, np.pi, -np.pi / 2]))

    def test_np_2(self):
        res = hfmath.angle(self.a, deg=True)
        self.assertAllclose(res, np.array([0, 90, 180, -90]))


class Test_make_matrix(TestCase):
    def test_make_matrix_1(self):
        a = hfarray([1, 10])
        b = hfarray([2, 20])
        c = hfarray([3, 30])
        d = hfarray([4, 40])
        i = DimMatrix_i("i", 2)
        j = DimMatrix_j("j", 2)
        fi = DimSweep("freq", [0, 1])
        matrix = hfmath.make_matrix(a, b, c, d)
        self.assertTrue(isinstance(matrix, hfarray))
        self.assertEqual(matrix.dims, (fi, i, j))
        self.assertAllclose(matrix, [[[1, 2], [3, 4]], [[10, 20], [30, 40]]])


class Test_chop(TestCase):
    def test_chop_1(self):
        res = hfmath.chop(np.array([0.123, 0.012, 0.00999]), 1e-2)
        self.assertAllclose(res, np.array([0.123, 0.012, 0]))

    def test_chop_2(self):
        self.assertEqual(hfmath.chop(0.123, 1e-2), 0.123)
        self.assertEqual(hfmath.chop(0.009999, 1e-2), 0.)


class Test_dB(TestCase):

    def test_1(self):
        self.assertEqual(hfmath.dB(.01 + 0j), -40)

    def test_2(self):
        self.assertEqual(hfmath.dB(100), 40)


class Test_dBinv(TestCase):

    def test_1(self):
        self.assertEqual(hfmath.dBinv(0), 1)

    def test_2(self):
        self.assertEqual(hfmath.dBinv(20), 10)

    def test_3(self):
        self.assertEqual(hfmath.dBinv(-20), 0.1)


class Test_dB_angle_to_complex(TestCase):
    dB = 0
    lin = 1

    def test_deg_0(self):
        res = hfmath.dB_angle_to_complex(self.dB, 0)
        self.assertAlmostEqual(res, 1 * self.lin)

    def test_deg_90(self):
        res = hfmath.dB_angle_to_complex(self.dB, 90)
        self.assertAlmostEqual(res, 1j * self.lin)

    def test_deg_180(self):
        res = hfmath.dB_angle_to_complex(self.dB, 180)
        self.assertAlmostEqual(res, -1 * self.lin)

    def test_deg_270(self):
        res = hfmath.dB_angle_to_complex(self.dB, 270)
        self.assertAlmostEqual(res, -1j * self.lin)


class Test_dB_angle_to_complex_20(Test_dB_angle_to_complex):
    dB = 20
    lin = 10


class Test_dB_angle_to_complex_neg_20(Test_dB_angle_to_complex):
    dB = -20
    lin = 0.1


class Test_mag_angle_to_complex(TestCase):
    lin = 1

    def test_deg_0(self):
        res = hfmath.mag_angle_to_complex(self.lin, 0)
        self.assertAlmostEqual(res, 1 * self.lin)

    def test_deg_90(self):
        res = hfmath.mag_angle_to_complex(self.lin, 90)
        self.assertAlmostEqual(res, 1j * self.lin)

    def test_deg_180(self):
        res = hfmath.mag_angle_to_complex(self.lin, 180)
        self.assertAlmostEqual(res, -1 * self.lin)

    def test_deg_270(self):
        res = hfmath.mag_angle_to_complex(self.lin, 270)
        self.assertAlmostEqual(res, -1j * self.lin)


class Test_mag_angle_to_complex_20(Test_mag_angle_to_complex):
    lin = 10


class Test_mag_angle_to_complex_neg_20(Test_mag_angle_to_complex):
    lin = 0.1


class Test_re_im_to_complex(TestCase):
    def test_re_im_1(self):
        self.assertAlmostEqual(hfmath.re_im_to_complex(0, 0), 0j)

    def test_re_im_2(self):
        self.assertAlmostEqual(hfmath.re_im_to_complex(1, 0), 1 + 0j)

    def test_re_im_3(self):
        self.assertAlmostEqual(hfmath.re_im_to_complex(1, 2), 1 + 2j)

    def test_re_im_4(self):
        self.assertAlmostEqual(hfmath.re_im_to_complex(-1, 2), -1 + 2j)

    def test_re_im_5(self):
        self.assertAlmostEqual(hfmath.re_im_to_complex(1, -2), 1 - 2j)

    def test_re_im_6(self):
        self.assertAlmostEqual(hfmath.re_im_to_complex(-1, -2), -1 - 2j)


class Test_continous_sqrt(TestCase):
    def test_1(self):
        a = hfarray(np.linspace(0, 6 * np.pi, 101))
        cplx = 4 * np.exp(1j * a)
        res = hfmath.continous_phase_sqrt(cplx)
        facit = 2 * np.exp(1j * a / 2)
        self.assertAllclose(res, facit)


class Test_unwrap_phase(TestCase):
    def test_1(self):
        a = hfarray(np.linspace(0, 6 * np.pi, 101))
        cplx = 4 * np.exp(1j * a)
        res = hfmath.unwrap_phase(cplx)
        facit = a
        self.assertAllclose(res, facit)

    def test_2(self):
        a = hfarray(np.linspace(0, 6 * np.pi, 101))
        cplx = 4 * np.exp(1j * a)
        res = hfmath.unwrap_phase(cplx, deg=True)
        facit = hfarray(np.linspace(0, 6 * 180, 101))
        self.assertAllclose(res, facit)


class Test_smooth(TestCase):
    data = [([1, 2, 1, 2, 1, 2.],
             2,
             np.array([1, 1.5, 1.5, 1.5, 1.5, 2])),
            (np.array([1, 2, 1, 2, 1.]),
             2,
             np.array([1, 1.5, 1.5, 1.5, 1])),
            (np.array([1, 2, 1, 2, 1.]),
             3,
             np.array([1, 4. / 3, 5. / 3, 4. / 3, 1])),
            ]

    def _help(self, data, aperture, facit):
        res = hfmath.smooth(data, aperture)
        self.assertEqual(res.shape, facit.shape)
        self.assertAllclose(res, facit)

    def test_0(self):
        self._help(*self.data[0])

    def test_1(self):
        self._help(*self.data[1])

    def test_2(self):
        self._help(*self.data[2])


class Test_smooth_mag_phase(TestCase):
    data = [(np.linspace(1, 5, 10) * np.exp(np.linspace(0, 1, 10) * 1j),
             2,
             (hfmath.smooth(np.linspace(1, 5, 10), 2) *
              np.exp(hfmath.smooth(np.linspace(0, 1, 10), 2) * 1j))),
            ]

    def _help(self, y, aperture, facit):
        res = hfmath.smooth_magphase(y, aperture=aperture)
        self.assertEqual(res.shape, facit.shape)
        self.assertAllclose(res, facit)

    def test_0(self):
        resdims = (DimSweep("freq", range(self.data[0][0].shape[0])),)
        y = hfarray(self.data[0][0], dims=resdims)
        self._help(y, *self.data[0][1:])


class Test_poly_smooth(TestCase):
    data = [(np.array([0, 1, 2, 3, 4, 5.]),
             np.array([0, 1, 4, 9, 16, 25.]), 4, 2,
             np.array([0, 1, 4, 9, 16, 25.])),
            (np.array([0, 1, 2, 3, 4, 5.]),
             np.array([0, 1, 4, 9, 16, 25.]), 4, 3,
             np.array([0, 1, 4, 9, 16, 25.])),
            (np.array([0, 1, 2, 3, 4, 5.]),
             np.array([0, 1, 4, 9, 16, 25.]), 5, 3,
             np.array([0, 1, 4, 9, 16, 25.])),
            ]

    def _help(self, x, y, aperture, N,  facit):
        res = hfmath.poly_smooth(x, y, aperture, N=N)
        self.assertEqual(res.shape, facit.shape)
        self.assertAllclose(res, facit)

    def test_0(self):
        self._help(*self.data[0])

    def test_1(self):
        self._help(*self.data[1])

    def test_2(self):
        self._help(*self.data[2])


class Test_linear_extrapolate(TestCase):
    def test_linear_extrapolate_1(self):
        res = hfmath.linear_extrapolate([1, 2, 3],
                                        [1, 4, 9],
                                        [0, 1., 1.5, 2.5, 3, 3.5])
        self.assertAllclose(res, [-2, 1, 2.5, 6.5, 9, 11.5])


class Test_interpolate(TestCase):
    def test_interpolate_1(self):
        res = hfmath.interpolate([1, 2, 3], [1, 4, 9], [1., 1.5, 2.5, 3])
        self.assertAllclose(res, [1, 2.5, 6.5, 9])

    def test_interpolate_to_low_error(self):
        self.assertRaises(ValueError, hfmath.interpolate, [1, 2, 3],
                          [1, 4, 9], [0.9, 1.5, 2.5, 3])

    def test_interpolate_to_high_error(self):
        self.assertRaises(ValueError, hfmath.interpolate, [1, 2, 3],
                          [1, 4, 9], [1, 1.5, 2.5, 3.1])


class Test_get_shape_helper(TestCase):
    def test_get_shape_helper_1(self):
        self.assertEqual(hfmath.get_shape_helper((1, 2, 3, 4), 1), (4,))

    def test_get_shape_helper_2(self):
        self.assertEqual(hfmath.get_shape_helper((1, 2, 3, 4), 2), (3, 4,))

    def test_get_shape_helper_3(self):
        self.assertEqual(hfmath.get_shape_helper((1, 2, 3, 4), 0), tuple())


class Test_get_dims_helper(TestCase):
    def test_get_dims_helper_1(self):
        self.assertEqual(hfmath.get_dims_helper((1, 2, 3, 4), 1), (1, 2, 3,))

    def test_get_dims_helper_2(self):
        self.assertEqual(hfmath.get_dims_helper((1, 2, 3, 4), 2), (1, 2,))

    def test_get_shape_helper_3(self):
        self.assertEqual(hfmath.get_dims_helper((1, 2, 3, 4), 0), tuple())


class Test_poly_smooth_mag_phase(TestCase):
    data = [(np.linspace(0, 10, 10),
             np.linspace(1, 5, 10) * np.exp(np.linspace(0, 1, 10) * 1j),
             4,
             2,
             np.linspace(1, 5, 10) * np.exp(np.linspace(0, 1, 10) * 1j)),
            ]

    def _help(self, x, y, aperture, N,  facit):
        res = hfmath.poly_smooth_magphase(x, y, aperture, N=N)
        self.assertEqual(res.shape, facit.shape)
        self.assertAllclose(res, facit)

    def test_0(self):
        resdims = (DimSweep("freq", self.data[0][0]),)
        y = hfarray(self.data[0][1], dims=resdims)
        self._help(self.data[0][0], y, *self.data[0][2:])


class TestInv(TestCase):
    def setUp(self):
        self.I = I = DimSweep("I", [1, 2])
        self.J = J = DimSweep("J", [10, 20, 30])
        self.K = K = DimSweep("K", [100, 200, 300, 400])
        self.mi = mi = DimMatrix_i("i", [0, 1])
        self.mj = mj = DimMatrix_j("j", [0, 1])
        self.i, self.j, self.k = (i, j, k) = map(hfarray, (I, J, K))

        self.a = i * j * k
        self.b = hfarray(i * j * k, unit="Hz", outputformat="%.3f")
        cdims = (DimMatrix_i("i", 2), DimMatrix_j("j", 2))
        self.c = hfarray([[11, 12], [21, 22]], dims=cdims)

        self.m = hfarray([[[1., 2], [3, 4]]] * 3, dims=(J, mi, mj))
        self.m2 = hfarray([[1., 2], [3, 4]], dims=(mi, mj))

    def test_1(self):
        res = hfmath.inv(self.m)
        self.assertAllclose(res, np.linalg.inv([[1., 2], [3, 4]]))
        self.assertEqual(res.shape, (3, 2, 2))
        self.assertEqual(res.dims, (self.J, self.mi, self.mj))


class TestMatrixMultiply(TestInv):
    def test_1(self):
        res = hfmath.matrix_multiply(self.m, self.m)
        self.assertAllclose(res, np.dot([[1., 2], [3, 4]], [[1., 2], [3, 4]]))
        self.assertEqual(res.shape, (3, 2, 2))
        self.assertEqual(res.dims, (self.J, self.mi, self.mj))

    def test_2(self):
        res = hfmath.matrix_multiply(self.m, self.m2)
        self.assertAllclose(res, np.dot([[1., 2], [3, 4]], [[1., 2], [3, 4]]))
        self.assertEqual(res.shape, (3, 2, 2))
        self.assertEqual(res.dims, (self.J, self.mi, self.mj))

    def test_3(self):
        res = hfmath.matrix_multiply(self.m2, self.m)
        self.assertAllclose(res, np.dot([[1., 2], [3, 4]], [[1., 2], [3, 4]]))
        self.assertEqual(res.shape, (3, 2, 2))
        self.assertEqual(res.dims, (self.J, self.mi, self.mj))

    def test_nparray(self):
        self.assertRaises(Exception, hfmath.matrix_multiply,
                          np.array(self.m), np.array(self.m))


class TestDet(TestInv):
    def test_1(self):
        res = hfmath.det(self.m)
        self.assertAllclose(res, -2)
        self.assertEqual(res.shape, (3, ))
        self.assertEqual(res.dims, (self.J, ))

if __name__ == '__main__':
    I = DimSweep("I", [1, 2])
    J = DimSweep("J", [10, 20, 30])
    K = DimSweep("K", [100, 200, 300, 400])
    mi = DimMatrix_i("i", [0, 1])
    mj = DimMatrix_j("j", [0, 1])
    (i, j, k) = map(hfarray, (I, J, K))

    a = i * j * k
    b = hfarray(i * j * k, unit="Hz", outputformat="%.3f")
    cdims = (DimMatrix_i("i", 2), DimMatrix_j("j", 2))
    c = hfarray([[11, 12], [21, 22]], dims=cdims)

    m = hfarray([[[1., 2], [3, 4]]] * 3, dims=(J, mi, mj))
    m2 = hfarray([[1., 2], [3, 4]], dims=(mi, mj))
