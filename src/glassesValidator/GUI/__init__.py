import sys
import pathlib

def run(project: str | pathlib.Path = None):

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

    if project is not None:
        from ._impl import utils
        if not utils.is_project_folder(project):
            raise ValueError(f'Project opening error: The selected folder ({project}) is not a project folder. Cannot open.')
        globals.project_path = pathlib.Path(project)
        
    from ._impl import async_thread, gui, process_pool
    async_thread.setup()
    globals.gui = gui.MainGUI()
    
    while True:
        # returns true if a new main_loop is needed, or false if all done
        should_restart = globals.gui.main_loop()

        process_pool.cleanup()

        if not should_restart:
            break

    # cleanup
    async_thread.cleanup()
    globals.cleanup()


