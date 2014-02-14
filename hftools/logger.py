# -*- coding: ISO-8859-1 -*-
#-----------------------------------------------------------------------------
# Copyright (c) 2014, HFTools Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import logging
import sys

logger = logging.getLogger("hftools")
logger.setLevel(logging.INFO)
_stream_handler = None


class NULLHandler(logging.Handler):
    def emit(self, s):
        pass
logger.addHandler(NULLHandler())


def start_log():
    """start_log starts logging of hftools channel to stderr
    """
    global _stream_handler
    if not _stream_handler:
        _stream_handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(_stream_handler)


def stop_log():
    """start_log stops logging of hftools channel to stderr
    """
    global _stream_handler
    if _stream_handler:
        logger.removeHandler(_stream_handler)
        _stream_handler = None
