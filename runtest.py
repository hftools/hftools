# -*- coding: utf-8 -*-
#
# DO NOT IMPORT ANYTHING FROM HFTOOLS HERE
# IMPORTING HFTOOLS AFFECTS COVERAGE MEASUREMENTs
from __future__ import print_function
import os
import sys

if sys.version_info[0] >= 3:
    import unittest
    configpath = "coverage-config-PY3.txt"
else:
    import unittest2 as unittest
    configpath = "coverage-config-PY2.txt"


pjoin = os.path.join
psplit = os.path.split

#warnings.filterwarnings(action='ignore', category=np.ModuleDeprecationWarning)
#import warnings
#warnings.filterwarnings(action='error', category=DeprecationWarning)


def main():
    omit = ["hftools/_external/*.py"]
    omit = None
    if "--coverage" in sys.argv:
        docoverage = True
        del sys.argv[sys.argv.index("--coverage")]
    else:
        docoverage = False

    if docoverage:
        try:
            import coverage
            cover_config = os.path.join(os.path.split(__file__)[0],
                                        configpath)
            cov = coverage.coverage(source=["hftools"],
                                    config_file=cover_config)
            cov.start()
        except ImportError:
            if docoverage:
                print("coverage module is missing. Running tests "
                      "without coverage")
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
