import cx_Freeze
import pathlib
import sys
import site
from distutils.util import convert_path

path = pathlib.Path(__file__).absolute().parent
sys.path.append(str(path/'src'))

def get_include_files():
    files = [path / "LICENSE"]

    # ffpyplayer bin deps
    for d in site.getsitepackages():
        d=pathlib.Path(d)/'share'/'ffpyplayer'
        for lib in ('ffmpeg', 'sdl'):
            d2 = d/lib/'bin'
            if d2.is_dir():
                for f in d2.iterdir():
                    if f.is_file() and f.suffix=='' or f.suffix in ['.dll', '.exe']:
                        files.append((f,pathlib.Path('lib')/f.name))
    return files

def get_zip_include_files():
    files = []
    todo = ['src/glassesValidator/config','src/glassesValidator/resources']
    for d in todo:
        d = pathlib.Path(convert_path(d))
        for d2 in d.iterdir():
            if d2.is_file() and d2.suffix not in ['.py','.pyc']:
                files.append((d2, pathlib.Path(*pathlib.Path(d2).parts[-3:])))
            elif not d2.name.startswith('__'):
                for f in d2.iterdir():
                    if f.is_file() and f.suffix not in ['.py','.pyc']:
                        files.append((f, pathlib.Path(*pathlib.Path(f).parts[-4:])))
    return files

main_ns = {}
ver_path = convert_path('src/glassesValidator/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

icon = convert_path('src/glassesValidator/resources/icons/icon')
if sys.platform.startswith("win"):
    icon += ".ico"
elif sys.platform.startswith("darwin"):
    icon += ".icns"
else:
    icon += ".png"

build_options = {
    "build_exe": {
        "optimize": 1,
        "packages": ['OpenGL','glassesValidator',
            'ffpyplayer.player','ffpyplayer.threading',      # some specific subpackages that need to be mentioned to be picked up correctly
            'imgui_bundle._imgui_bundle'
        ],
        "excludes":["tkinter"],
        "zip_includes": get_zip_include_files(),
        "zip_include_packages": "*",
        "zip_exclude_packages": [
            "OpenGL_accelerate",
            "glfw",
            "imgui_bundle",
        ],
        "silent_level": 1,
        "include_msvcr": True
    },
    "bdist_mac": {
        "bundle_name": "glassesValidator",
        "iconfile": icon,
        "codesign_identity": None,
        "plist_items": [
            ("CFBundleName", "glassesValidator"),
            ("CFBundleDisplayName", "glassesValidator"),
            ("CFBundleIdentifier", "com.github.dcnieho.glassesValidator"),
            ("CFBundleVersion", main_ns['__version__']),
            ("CFBundlePackageType", "APPL"),
            ("CFBundleSignature", "????"),
        ]
    }
}
if sys.platform.startswith("win"):
    build_options["build_exe"]["include_files"] = get_include_files()

cx_Freeze.setup(
    name="glassesValidator",
    version=main_ns['__version__'],
    description=main_ns['__description__'],
    executables=[
        cx_Freeze.Executable(
            script=path / "main.py",
            target_name="glassesValidator",
            icon=icon
        )
    ],
    options=build_options,
    py_modules=[]
)
