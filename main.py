import sys
import pathlib

frozen = getattr(sys, "frozen", False)

if frozen:
    self_path = pathlib.Path(sys.executable).parent
else:
    self_path = pathlib.Path(__file__).parent
    sys.path.append(str(self_path/"src"))

import glassesValidator

glassesValidator.GUI.run()
