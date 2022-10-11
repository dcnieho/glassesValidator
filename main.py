#!/usr/bin/env python
import sys

if getattr(sys, "frozen", False):
    # need to call this so that code in __init__ of ffpyplayer
    # doesn't encounter a None in site.USER_BASE
    import site
    site.getuserbase()  
else:
    import pathlib
    src_path = str(pathlib.Path(__file__).parent/"src")
    if not src_path in sys.path:
        sys.path.append(src_path)

if __name__=="__main__":
    import glassesValidator
    
    glassesValidator.GUI.run()