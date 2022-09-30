import pathlib
import sys
import os

frozen = getattr(sys, "frozen", False)

if frozen and sys.platform.startswith("linux"):
    self_path = pathlib.Path(sys.executable).parent
    library = self_path / f"lib/glfw/{os.environ.get('XDG_SESSION_TYPE')}/libglfw.so"
    if library.is_file():
        os.environ["PYGLFW_LIBRARY"] = str(library)
       
from ... import __version__
version           = __version__
pypi_page         = "https://pypi.org/project/glassesValidator/"
github_page       = "https://github.com/dcnieho/glassesValidator"
developer_page    = "https://www.humlab.lu.se/person/DiederickCNiehorster/"
reference         = "Niehorster, D.C., Hessels, R.S., Benjamins, J.S., Nyström, M. & Hooge, I.T.C. (in prep). GlassesValidator: Data quality tool for eye tracking glasses."
reference_bibtex  = "<bibtex here>"


from .structs import CounterContext, JobDescription, Os, Settings
from .gui import MainGUI
from ...utils import Recording

home = pathlib.Path.home()
if sys.platform.startswith("win"):
    os = Os.Windows
    data_path = "AppData/Roaming/glassesValidator"
elif sys.platform.startswith("linux"):
    os = Os.Linux
    data_path = ".config/glassesValidator"
elif sys.platform.startswith("darwin"):
    os = Os.MacOS
    data_path = "Library/Application Support/glassesValidator"
else:
    print("Your system is not officially supported at the moment!\n"
          "You can let me know on the tool thread or on GitHub, or you can try porting yourself ;)")
    sys.exit(1)
data_path = home / data_path
data_path.mkdir(parents=True, exist_ok=True)

# Variables
popup_stack = []
project_path: pathlib.Path = None
should_exit: bool = False
gui: MainGUI = None
settings: Settings = None
rec_id: CounterContext = CounterContext()
recordings: dict[int, Recording] = None
selected_recordings: dict[int, bool] = None
jobs: dict[int, JobDescription] = None