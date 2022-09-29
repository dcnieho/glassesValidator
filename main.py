#!/usr/bin/env python
import sys
import pathlib

frozen = getattr(sys, "frozen", False)

if frozen:
    self_path = pathlib.Path(sys.executable).parent
else:
    self_path = pathlib.Path(__file__).parent
    src_path  = str(self_path/"src")
    if not src_path in sys.path:
        sys.path.append(src_path)

if __name__=="__main__":
    import glassesValidator
    
    glassesValidator.GUI.run()