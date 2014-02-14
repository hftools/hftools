# -*- coding: utf-8 -*-
import unittest2 as unittest
import os
import sys
import warnings
import numpy as np
from fnmatch import fnmatch
from unittest2.main import TestProgram

pjoin = os.path.join
psplit = os.path.split

#warnings.filterwarnings(action='ignore', category=np.ModuleDeprecationWarning)
#warnings.filterwarnings(action='error', category=DeprecationWarning)



def main():
    omit = ["hftools/_external/*.py"]
    if "--coverage" in sys.argv:
        docoverage = True
        del sys.argv[sys.argv.index("--coverage")]
    else:
        docoverage = False
    for root, dirs, files in os.walk('.'):
        omit.extend(os.path.join(root, f) for f in files
                    if os.path.split(root)[1] in ["tests", "testing"])
    for root, dirs, files in os.walk(os.path.split(__file__)[0]):
        omit.extend(os.path.join(root, f) for f in files
                    if os.path.split(root)[1] in ["tests", "testing"])

    omit = []

    if docoverage:
        try:
            import coverage
            cover_config = os.path.join(os.path.split(__file__)[0],
                                        "coverage-config.txt")
            cov = coverage.coverage(source=["hftools"], omit=omit,
                                    config_file=cover_config)
            cov.start()
        except ImportError:
            if docoverage:
                print "coverage module is missing. Running tests "\
                      "without coverage"
            docoverage = False

    runner = unittest.TextTestRunner  # OBS do not instantiate class!
    prog = unittest.TestProgram(module=None, testRunner=runner, exit=False)

    if docoverage:
        cov.stop()
        import hftools
        covhtmldir = pjoin(psplit(psplit(hftools.__file__)[0])[0], "covhtml")
        cov.html_report(directory=covhtmldir)

if __name__ == '__main__':
    main()
