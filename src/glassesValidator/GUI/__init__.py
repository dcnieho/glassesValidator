import sys
import multiprocessing

def run():
    multiprocessing.freeze_support()

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
        
    from ._impl import async_thread, gui, process_pool
    async_thread.setup()
    globals.gui = gui.MainGUI()
    
    while True:
        process_pool.setup()

        # returns true if a new main_loop is needed, or false if all done
        should_restart = globals.gui.main_loop()

        process_pool.cleanup()

        if not should_restart:
            break


