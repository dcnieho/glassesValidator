#!/usr/bin/env python
import sys
import os
import multiprocessing

if getattr(sys, "frozen", False):
    if not sys.platform.startswith("win"):
        raise "Executable is only supported on Windows"

    # need to call this so that code in __init__ of ffpyplayer
    # doesn't encounter a None in site.USER_BASE
    import site
    site.getuserbase()

    # need to put packaged ffmpeg executable on path
    p = os.path.join(os.path.dirname(sys.executable),'lib')
    os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
    os.add_dll_directory(p)

if __name__=="__main__":
    multiprocessing.freeze_support()

    import glassesValidator
    glassesValidator.GUI.run()
