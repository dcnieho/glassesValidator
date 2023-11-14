# -*- coding: utf-8 -*-
import pathlib
from enum import Enum, auto
import pandas as pd

from .b_codeMarkerInterval import process as code_marker_interval
from .c_detectMarkers import process as detect_markers
from .d_gazeToPoster import process as gaze_to_poster
from .e1_computeOffsetsToTargets import process as compute_offsets_to_targets
from .e2_determineFixationIntervals import process as determine_fixation_intervals
from .f_calculateDataQuality import process as calculate_data_quality

# expose code_marker_interval (stage 2 in our 3-step process) under a simpler name
# NB: not a simple alias as we're hiding the third input argument for simple use
def do_coding(working_dir: str | pathlib.Path, config_dir=None):
    return code_marker_interval(working_dir, config_dir)

# package the further steps in a single function to simplify using this (i.e. group into a single step 3)
def do_process(working_dir: str | pathlib.Path, config_dir=None):
    detect_markers(working_dir, config_dir)
    gaze_to_poster(working_dir, config_dir)
    compute_offsets_to_targets(working_dir, config_dir)
    determine_fixation_intervals(working_dir, config_dir)
    calculate_data_quality(working_dir, config_dir)



# NB: using pose information requires a calibrated scene camera
class DataQualityType(Enum):
    viewpos_vidpos_homography   = auto()    # use homography to map gaze from video to poster, and viewing distance defined in config (combined with the assumptions that the viewing position (eye) is located directly in front of the poster's center and that the poster is oriented perpendicularly to the line of sight) to compute angular measures
    pose_vidpos_homography      = auto()    # use homography to map gaze from video to poster, and pose information w.r.t. poster to compute angular measures
    pose_vidpos_ray             = auto()    # use camera calibration to map gaze postion on scene video to cyclopean gaze vector, and pose information w.r.t. poster to compute angular measures
    pose_world_eye              = auto()    # use provided gaze position in world (often a binocular gaze point), and pose information w.r.t. poster to compute angular measures
    pose_left_eye               = auto()    # use provided left eye gaze vector, and pose information w.r.t. poster to compute angular measures
    pose_right_eye              = auto()    # use provided right eye gaze vector, and pose information w.r.t. poster to compute angular measures
    pose_left_right_avg         = auto()    # report average of left (pose_left_eye) and right (pose_right_eye) eye angular values

    # so this get serialized in a user-friendly way by pandas..
    def __str__(self):
        return self.name

def get_DataQualityType_explanation(dq: DataQualityType):
    ler_name =  "Left eye ray + pose"
    rer_name = "Right eye ray + pose"
    match dq:
        case DataQualityType.viewpos_vidpos_homography:
            return "Homography + view distance", \
                   "Use homography to map gaze position from the scene video to " \
                   "the validation poster, and use an assumed viewing distance (see " \
                   "the project's configuration) to compute data quality measures " \
                   "in degrees with respect to the scene camera. In this mode, it is "\
                   "assumed that the eye is located exactly in front of the center of "\
                   "the poster and that the poster is oriented perpendicularly to the "\
                   "line of sight from this assumed viewing position."
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
        case DataQualityType.pose_world_eye:
            return "World gaze position + pose", \
                   "Use the gaze position in the world provided by the eye tracker " \
                   "(often a binocular gaze point) to determine gaze position on the " \
                   "validation poster by turning it into a direction vector with respect " \
                   "to the scene camera and intersecting this vector with the poster. " \
                   "Then, use the determined pose of the scene camera " \
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


def _collect_data_quality(rec_dirs: list[str | pathlib.Path]):
    # 1. collect all data quality from the selected
    rec_files = [pathlib.Path(rec)/'dataQuality.tsv' for rec in rec_dirs]
    rec_files = [f for f in rec_files if f.is_file()]
    if not rec_files:
        return None, None, None
    df = pd.concat((pd.read_csv(rec, delimiter='\t').assign(recording=rec.parent.name) for rec in rec_files), ignore_index=True)
    if df.empty:
        return None, None, None
    # set indices
    df = df.set_index(['recording','marker_interval','type','target'])
    # change type index into enum
    typeIdx = df.index.names.index('type')
    df.index = df.index.set_levels(pd.CategoricalIndex([getattr(DataQualityType,x) for x in df.index.levels[typeIdx]]),level='type')

    # see what we have
    dq_types = sorted(list(df.index.levels[typeIdx]), key=lambda dq: dq.value)
    targets  = list(df.index.levels[df.index.names.index('target')])

    # good default selection of dq type to export
    if DataQualityType.pose_vidpos_ray in dq_types:
        default_dq_type = DataQualityType.pose_vidpos_ray
    elif DataQualityType.pose_vidpos_homography in dq_types:
        default_dq_type = DataQualityType.pose_vidpos_homography
    else:
        # ultimate fallback, just set first available as the one to export
        default_dq_type = dq_types[0]

    return df, default_dq_type, targets

def _summarize_and_store_data_quality(df: pd.DataFrame, output_file_or_dir: str | pathlib.Path, dq_types: list[DataQualityType], targets: list[int], average_over_targets = False, include_data_loss = False):
    dq_types_have = sorted(list(df.index.levels[df.index.names.index('type')]), key=lambda dq: dq.value)
    targets_have  = list(df.index.levels[df.index.names.index('target')])

    # remove unwanted types of data quality
    dq_types_sel = [dq in dq_types for dq in dq_types_have]
    if not all(dq_types_sel):
        df = df.drop(index=[dq for i,dq in enumerate(dq_types_have) if not dq_types_sel[i]], level='type')
    # remove unwanted targets
    targets_sel = [t in targets for t in targets_have]
    if not all(targets_sel):
        df = df.drop(index=[t for i,t in enumerate(targets_have) if not targets_sel[i]], level='target')
    # remove unwanted data loss
    if not include_data_loss and 'data_loss' in df.columns:
        df = df.drop(columns='data_loss')
    # average data if wanted
    if average_over_targets:
        gb = df.drop(columns='order').groupby(['recording', 'marker_interval', 'type'],observed=True)
        count = gb.count()
        df = gb.mean()
        # add number of targets count (there may be some missing data)
        df.insert(0,'num_targets',count['acc'])

    # store
    output_file_or_dir = pathlib.Path(output_file_or_dir)
    if output_file_or_dir.is_dir():
        output_file_or_dir = output_file_or_dir / 'dataQuality.tsv'
    df.to_csv(str(output_file_or_dir), mode='w', header=True, sep='\t', na_rep='nan', float_format="%.6f")

def export_data_quality(rec_dirs: list[str | pathlib.Path], output_file_or_dir: str | pathlib.Path, dq_types: list[DataQualityType] = None, targets: list[int] = None, average_over_targets = False, include_data_loss = False):
    df, default_dq_type, targets_have = _collect_data_quality(rec_dirs)
    if not dq_types:
        dq_types = [default_dq_type]
    if not targets:
        targets = targets_have
    _summarize_and_store_data_quality(df, output_file_or_dir, dq_types, targets, average_over_targets, include_data_loss)


__all__ = ['code_marker_interval','detect_markers','gaze_to_poster',
           'compute_offsets_to_targets','determine_fixation_intervals','calculate_data_quality',
           'do_coding','do_process','DataQualityType','export_data_quality']