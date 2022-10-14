#!/usr/bin/env python
import sys
import multiprocessing

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
    # on Windows, multiprocessing.freeze_support() takes care of not executing
    # the code below when frozen, but instead the correct child process code.
    # But multiprocessing.freeze_support() is Windows-only. On MacOS and Linux,
    # we emulate its behavior ourselves, when needed. See
    # https://github.com/python/cpython/pull/5195,
    # https://github.com/python/cpython/issues/76327, and
    # https://stackoverflow.com/a/47360452
    multiprocessing.freeze_support()
    if not sys.platform.startswith("win") and getattr(sys, "frozen", False) and len(sys.argv)>=3:
        # test if this is the main process or a spawned child
        if sys.argv[1]=='--multiprocessing-fork' or \
           'from multiprocessing.resource_tracker import main' in sys.argv[-1] \
           :
            # we're a multiprocessing child process, now see what the call is and route it to the right function
            if 'from multiprocessing.resource_tracker import main' in sys.argv[-1]:
                from multiprocessing.resource_tracker import main
                
                # command ends with main(fd) - extract the fd.
                fd = int(sys.argv[-1].rsplit('main(')[1].split(')')[0])
                main(fd)
            else:
                from multiprocessing.spawn import spawn_main
                
                kwds = {k:int(v) for arg in sys.argv[2:] for k,v in [arg.split('=', maxsplit=1)]}
                spawn_main(**kwds)
            
            # child process done, exit
            sys.exit()
        
    # normal main process code to run
    import glassesValidator
    
    glassesValidator.GUI.run()
