# -*- coding: utf-8 -*-
import pathlib

from .b_codeMarkerInterval import process as codeMarkerInterval
from .c_detectMarkers import process as detectMarkers
from .d_gazeToBoard import process as gazeToBoard
from .e1_computeOffsetsToTargets import process as computeOffsetsToTargets
from .e2_determineFixationIntervals import process as determineFixationIntervals
from .f_calculateDataQuality import process as calculateDataQuality

# expose codeMarkerInterval (stage 2 in our 3-step process) under a simpler name
# NB: not a simple alias as we're hiding the third input argument for simple use
def do_coding(folder: str | pathlib.Path, config_dir=None):
    return codeMarkerInterval(folder, config_dir)

# package the further steps in a single function to simplify using this (i.e. group into a single step 3)
def do_process(folder: str | pathlib.Path, config_dir=None):
    detectMarkers(folder, config_dir)
    gazeToBoard(folder, config_dir)
    computeOffsetsToTargets(folder, config_dir)
    determineFixationIntervals(folder, config_dir)
    calculateDataQuality(folder, config_dir)


__all__ = ['codeMarkerInterval','detectMarkers','gazeToBoard',
           'computeOffsetsToTargets','determineFixationIntervals','calculateDataQuality',
           'do_coding','do_process']