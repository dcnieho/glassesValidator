
from shlex import shlex
import shutil
import pandas as pd
import numpy as np
import pathlib
import importlib.resources

from . import markerBoard

def _readValidationSetupFile(file):
    # read key=value pairs into dict
    lexer = shlex(file)
    lexer.whitespace += '='
    lexer.wordchars += '.[],'   # don't split extensions of filenames in the input file, and accept stuff from python list syntax
    lexer.commenters = '%'
    return dict(zip(lexer, lexer))

def getValidationSetup(configDir=None, setupFile='validationSetup.txt'):
    if configDir is not None:
        with (pathlib.Path(configDir) / setupFile).open() as f:
            validationSetup = _readValidationSetupFile(f)
    else:
        # fall back on default config included with package
        with importlib.resources.open_text('glassesValidator.config', setupFile) as f:
            validationSetup = _readValidationSetupFile(f)

    # parse numerics into int or float
    for key,val in validationSetup.items():
        if np.all([c.isdigit() for c in val]):
            validationSetup[key] = int(val)
        else:
            try:
                validationSetup[key] = float(val)
            except:
                pass # just keep value as a string
    return validationSetup

def _readCoordFile(file):
    return pd.read_csv(str(file),index_col=0)

def _getCoordFile(configDir, file):
    if configDir is not None:
        if (configDir / file).is_file():
            return _readCoordFile(configDir / file)
        else:
            return None
    else:
        with importlib.resources.path('glassesValidator.config', file) as p:
            return _readCoordFile(p)

def getTargets(configDir=None, file='targetPositions.csv'):
    return _getCoordFile(configDir, file)

def getMarkers(configDir=None, file='markerPositions.csv'):
    return _getCoordFile(configDir, file)

            

def deployValidationConfig(outDir):
    outDir = pathlib.Path(outDir)
    if not outDir.is_dir():
        raise RuntimeError('the requested directory "%s" does not exist' % outDir)

    # copy over all config files
    for r in ['markerPositions.csv', 'targetPositions.csv', 'validationSetup.txt']:
        with importlib.resources.path('glassesValidator.config', r) as p:
            shutil.copyfile(p, str(outDir/r))

    # copy over markerBoard text file
    boardDir = outDir / 'markerBoard'
    if not boardDir.is_dir():
        boardDir.mkdir()

    markerBoard.deployMaker(boardDir)