#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import numpy as np
import os, pdb

from hftools.testing import TestCase

class Test_1(TestCase):
    readfun = None
    basepath = None
    dirname = None
    extension = None
    filename = "test1"
    readpars = {"verbose":False}
    def setUp(self):
        self.facit_f = np.array([0, 1, 2, 3]) * 1e9
        self.facit_s = (np.array([[[1, 2], [3, 4]]]) *
                        np.array([1, 1j, -1, -1j])[:, np.newaxis, np.newaxis])
        self.comments = []
        self.read_data()

    def read_data(self):
        fname = self.basepath / u"testdata" / self.dirname / self.filename + self.extension
        self.block = self.readfun[0](fname, **self.readpars)

    def test_comments(self):
        self.assertEqual(self.block.comments.fullcomments, self.comments)

    def test_freq(self):
        self.assertAllclose(self.facit_f, self.block["freq"])

    def test_data(self):
        self.assertAllclose(self.facit_s, self.block["S"])

class Test_2(Test_1):
    filename = "test2"

class Test_3(Test_1):
    filename = "test3"


class Test_Comment_1(Test_1):
    filename = "test4"
    readpars = dict(property_to_vars=True, verbose=False)
    def setUp(self):
        Test_1.setUp(self)
        self.comments = ["Vgs [V]= -1",
                         "Vds [V]= 5.3",
                         "Ig [uA]= 0.3",
                         "Id [mA]= 20",
                        ]
    def test_prop(self):
        self.assertAllclose(self.block["Vgs"], -1)
        self.assertAllclose(self.block["Vds"], 5.3)
        self.assertAllclose(self.block["Ig"], 0.3e-6)
        self.assertAllclose(self.block["Id"], 0.02)

class Test_Comment_2(Test_1):
    filename = "test4"
    readpars = dict(property_to_vars=False, verbose=False)
    def setUp(self):
        Test_1.setUp(self)
        self.comments = ["Vgs [V]= -1",
                         "Vds [V]= 5.3",
                         "Ig [uA]= 0.3",
                         "Id [mA]= 20",
                        ]
    def test_prop(self):
        self.assertRaises(KeyError, lambda x:self.block[x], "Vgs", )
        self.assertRaises(KeyError, lambda x:self.block[x], "Vds", )
        self.assertRaises(KeyError, lambda x:self.block[x], "Ig", )
        self.assertRaises(KeyError, lambda x:self.block[x], "Id", )

