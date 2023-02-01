
from shlex import shlex
import shutil
import pandas as pd
import numpy as np
import pathlib
import importlib.resources

from . import poster

def _readValidationSetupFile(file):
    # read key=value pairs into dict
    lexer = shlex(file)
    lexer.whitespace += '='
    lexer.wordchars += '.[],'   # don't split extensions of filenames in the input file, and accept stuff from python list syntax
    lexer.commenters = '%'
    return dict(zip(lexer, lexer))

def get_validation_setup(config_dir=None, setup_file='validationSetup.txt'):
    if config_dir is not None:
        with (pathlib.Path(config_dir) / setup_file).open() as f:
            validationSetup = _readValidationSetupFile(f)
    else:
        # fall back on default config included with package
        with importlib.resources.open_text(__package__, setup_file) as f:
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

def _getCoordFile(config_dir, file):
    if config_dir is not None:
        if (config_dir / file).is_file():
            return _readCoordFile(config_dir / file)
        else:
            return None
    else:
        with importlib.resources.path(__package__, file) as p:
            return _readCoordFile(p)

def get_targets(config_dir=None, file='targetPositions.csv'):
    return _getCoordFile(config_dir, file)

def get_markers(config_dir=None, file='markerPositions.csv'):
    return _getCoordFile(config_dir, file)



def deploy_validation_config(output_dir):
    output_dir = pathlib.Path(output_dir)
    if not output_dir.is_dir():
        raise RuntimeError('the requested directory "%s" does not exist' % output_dir)

    # copy over all config files
    for r in ['markerPositions.csv', 'targetPositions.csv', 'validationSetup.txt']:
        with importlib.resources.path(__package__, r) as p:
            shutil.copyfile(p, str(output_dir/r))

    # copy over poster tex file
    poster_dir = output_dir / 'poster'
    if not poster_dir.is_dir():
        poster_dir.mkdir()

    poster.deploy_maker(poster_dir)