import pathlib
import typing
from imgui_bundle import imgui

from . import globals




def get_data_path():
    if globals.project_path is not None:
        return globals.project_path
    else:
        return globals.data_path

def is_project_folder(folder: str | pathlib.Path):
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        return False
    # a project directory should contain the (empty)
    # glassesValidator.project file and imgui.ini file
    return (folder/'imgui.ini').is_file() and (folder/'glassesValidator.project').is_file()


def init_project_folder(folder: str | pathlib.Path, imgui_ini_saver: typing.Callable = None):
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        return
    # a project directory should contain the empty
    # glassesValidator.project file, so we create it
    # here.
    # Also, a copy of imgui.ini to persist some settings
    # of the current shown GUI is good to have, so save it here.

    if imgui_ini_saver is not None:
        imgui_ini_saver(folder/'imgui.ini')
    else:
        imgui.save_ini_settings_to_disk(str(folder/'imgui.ini'))
    (folder/'glassesValidator.project').touch()
