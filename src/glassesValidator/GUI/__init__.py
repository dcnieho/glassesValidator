import sys
import pathlib

def run(project_dir: str | pathlib.Path = None):
    from glassesTools import platform
    from ._impl import globals

    if platform.os is platform.Os.Windows:
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

    if project_dir is not None:
        from ._impl import utils
        if not utils.is_project_folder(project_dir):
            raise ValueError(f'Project opening error: The selected folder ({project_dir}) is not a project folder. Cannot open.')
        globals.project_path = pathlib.Path(project_dir)

    from ._impl import gui, process_pool
    from glassesTools import async_thread
    async_thread.setup()
    globals.gui = gui.MainGUI()

    globals.gui.run()
    process_pool.cleanup()

    # cleanup
    async_thread.cleanup()
    globals.cleanup()

