# -*- coding: utf-8 -*-
import pathlib
from enum import Enum, auto

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


# NB: using pose information requires a calibrated scene camera
class DataQualityType(Enum):
    viewdist_vidpos_homography  = auto()    # use homography to map gaze from video to marker board, and viewing distance defined in config to compute angular measures
    pose_vidpos_homography      = auto()    # use homography to map gaze from video to marker board, and pose information w.r.t. marker board to compute angular measures
    pose_vidpos_ray             = auto()    # use camera calibration to map gaze postion in scene video to cyclopean gaze vector, and pose information w.r.t. marker board to compute angular measures
    pose_left_eye               = auto()    # use provided left eye gaze vector, and pose information w.r.t. marker board to compute angular measures
    pose_right_eye              = auto()    # use provided right eye gaze vector, and pose information w.r.t. marker board to compute angular measures
    pose_left_right_avg         = auto()    # report average of left (pose_left_eye) and right (pose_right_eye) eye angular values
    
    # so this get serialized in a user-friendly way by pandas..
    def __str__(self):
        return self.name

def get_DataQualityType_explanation(dq: DataQualityType):
    ler_name =  "Left eye ray + pose"
    rer_name = "Right eye ray + pose"
    match dq:
        case DataQualityType.viewdist_vidpos_homography:
            return "Homography + view distance", \
                   "Use homography to map gaze position from the scene video to " \
                   "the validation poster, and use an assumed viewing distance (see " \
                   "the project's configuration) to compute data quality measures " \
                   "in degrees with respect to the scene camera."
        case DataQualityType.pose_vidpos_homography:
            return "Homography + pose", \
                   "Use homography to map gaze position from the scene video to " \
                   "the validation poster, and use the determined pose of the scene " \
                   "camera (requires a calibrated camera) to compute data quality " \
                   "measures in degrees with respect to the scene camera."
        case DataQualityType.pose_vidpos_ray:
            return "Video ray + pose", \
                   "Use camera calibration to turn gaze position from the scene " \
                   "video into a direction vector, and determine gaze position on " \
                   "the validation poster by intersecting this vector with the " \
                   "poster. Then, use the determined pose of the scene camera " \
                   "(requires a calibrated camera) to compute data quality " \
                   "measures in degrees with respect to the scene camera."
        case DataQualityType.pose_left_eye:
            return ler_name, \
                   "Use the gaze direction vector for the left eye provided by " \
                   "the eye tracker to determine gaze position on the validation " \
                   "poster by intersecting this vector with the poster. " \
                   "Then, use the determined pose of the scene camera " \
                   "(requires a camera calibration) to compute data quality " \
                   "measures in degrees with respect to the left eye."
        case DataQualityType.pose_right_eye:
            return rer_name, \
                   "Use the gaze direction vector for the right eye provided by " \
                   "the eye tracker to determine gaze position on the validation " \
                   "poster by intersecting this vector with the poster. " \
                   "Then, use the determined pose of the scene camera " \
                   "(requires a camera calibration) to compute data quality " \
                   "measures in degrees with respect to the right eye."
        case DataQualityType.pose_left_right_avg:
            return "Average eye rays + pose", \
                   "For each time point, take angular offset between the left and " \
                   "right gaze positions and the fixation target and average them " \
                   "to compute data quality measures in degrees. Requires " \
                   f"'{ler_name}' and '{rer_name}' to be enabled."


__all__ = ['codeMarkerInterval','detectMarkers','gazeToBoard',
           'computeOffsetsToTargets','determineFixationIntervals','calculateDataQuality',
           'do_coding','do_process','DataQualityType']