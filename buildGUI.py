from ctypes.util import find_library
import cx_Freeze
import pathlib
import sys
from distutils.util import convert_path

sys.setrecursionlimit(1500)

path = pathlib.Path(__file__).absolute().parent
sys.path.append(str(path/'src'))

main_ns = {}
ver_path = convert_path('src/glassesValidator/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

cx_Freeze.setup(
    name="glassesValidator",
    version=main_ns['__version__'],
    description=main_ns['__description__'],
    executables=[
        cx_Freeze.Executable(
            script=path / "main.py",
            target_name="glassesValidator",
        )
    ],
    options={
        "build_exe": {
            "optimize": 1,
            "packages": [
                'numpy','matplotlib','scipy','pandas','glassesValidator'
            ],
            #"includes": ["glassesValidator"],
            "include_files": [
                path / "resources",
                path / "LICENSE"
            ],
            "zip_include_packages": "*",
            "zip_exclude_packages": [
                "OpenGL_accelerate",
                "PyQt6",
                "glfw"
            ],
            "silent_level": 1,
            "include_msvcr": True
        },
        "bdist_mac": {
            "bundle_name": "glassesValidator",
            "plist_items": [
                ("CFBundleName", "glassesValidator"),
                ("CFBundleDisplayName", "glassesValidator"),
                ("CFBundleIdentifier", "com.github.dcnieho.glassesValidator"),
                ("CFBundleVersion", main_ns['__version__']),
                ("CFBundlePackageType", "APPL"),
                ("CFBundleSignature", "????"),
            ]
        }
    },
    py_modules=[]
)
