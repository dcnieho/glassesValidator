# -*- coding: utf-8 -*-
import pathlib

from .b_codeMarkerInterval import process as code_marker_interval
from .c_detectMarkers import process as detect_markers
from .d_gazeToPlane import process as gaze_to_plane
from .e_classifyFixations import process as classify_fixations
from .f_assignFixations import process as assign_fixations
from .g_calculateDataQuality import process as calculate_data_quality

# expose code_marker_interval (stage 2 in our 3-step process) under a simpler name
# NB: not a simple alias as we're hiding the third input argument for simple use
def do_coding(working_dir: str | pathlib.Path, config_dir=None):
    return code_marker_interval(working_dir, config_dir)

# package the further steps in a single function to simplify using this (i.e. group into a single step 3)
def do_process(working_dir: str | pathlib.Path, config_dir=None):
    detect_markers(working_dir, config_dir)
    gaze_to_plane(working_dir, config_dir)
    classify_fixations(working_dir, config_dir)
    assign_fixations(working_dir, config_dir)
    calculate_data_quality(working_dir, config_dir)


__all__ = ['code_marker_interval','detect_markers','gaze_to_plane',
           'classify_fixations','assign_fixations','calculate_data_quality',
           'do_coding','do_process']