# -*- coding: utf-8 -*-

import sys
if sys.platform.startswith("linux"):
    # something imported later in glassesValidator causes opencv windows not to show
    # unless one was already briefly created (not just cv2 imported first), on Linux.
    # so below is a shitty patch making things work.
    import cv2
    cv2.namedWindow("dummy",cv2.WINDOW_NORMAL)
    cv2.destroyAllWindows()

from . import config
from . import GUI
from . import preprocess
from . import process
from . import utils
from .version import __version__, __url__, __author__, __email__, __description__
