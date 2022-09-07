#!/usr/bin/env python
import sys


def run():
    from ._impl import globals

    from ._impl.structs import Os


    if globals.os is Os.Windows:
        # Hide conhost if frozen or release
        if globals.frozen and "nohide" not in sys.argv:
            import ctypes
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
    else:
        # Install uvloop on MacOS and Linux
        try:
            import uvloop
            uvloop.install()
        except Exception:
            pass

        
    from ._impl import async_thread, sync_thread
    async_thread.setup()
    sync_thread.setup()
    
    from ._impl import db
    from ._impl import gui
    db.setup()

    globals.gui = gui.MainGUI()

    # only returns once GUI is closed, may need to refresh to open another project
    globals.gui.main_loop()

    # tear down
    db.shutdown()
