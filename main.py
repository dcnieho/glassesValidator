#!/usr/bin/env python
import sys
import multiprocessing

if getattr(sys, "frozen", False):
    if not sys.platform.startswith("win"):
        raise "Executable is only supported on Windows"

    # need to call this so that code in __init__ of ffpyplayer
    # doesn't encounter a None in site.USER_BASE
    import site
    site.getuserbase()

if __name__=="__main__":
    multiprocessing.freeze_support()

    import glassesValidator
    glassesValidator.GUI.run()
