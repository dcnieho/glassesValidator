#!/usr/bin/env python
import sys

if getattr(sys, "frozen", False):
    raise RuntimeError('No freeze support for this code')

if __name__=="__main__":
    import glassesValidator
    glassesValidator.GUI.run()
