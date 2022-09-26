# -*- coding: utf-8 -*-
import pathlib
import json

from .. import utils as _utils

from .tobii_G2 import preprocessData as tobii_G2
from .tobii_G3 import preprocessData as tobii_G3
from .SMI_ETG import preprocessData as SMI_ETG
from .SeeTrue import preprocessData as SeeTrue

def pupil_core(output_dir: str | pathlib.Path, source_dir: str | pathlib.Path = None, recInfo: _utils.Recording = None):
    from .pupilLabs import preprocessData
    preprocessData(output_dir, 'Pupil Core', source_dir, recInfo)


def pupil_invisible(output_dir: str | pathlib.Path, source_dir: str | pathlib.Path = None, recInfo: _utils.Recording = None):
    from .pupilLabs import preprocessData
    preprocessData(output_dir, 'Pupil Invisible', source_dir, recInfo)

    
def get_recording_info(source_dir: str | pathlib.Path, device: str | _utils.EyeTracker):
    source_dir  = pathlib.Path(source_dir)
    device = _utils.type_string_to_enum(device)

    rec_info = None
    match device:
        case _utils.EyeTracker.Pupil_Core:
            from .pupilLabs import getRecordingInfo
            rec_info = getRecordingInfo(source_dir, device)
        case _utils.EyeTracker.Pupil_Invisible:
            from .pupilLabs import getRecordingInfo
            rec_info = getRecordingInfo(source_dir, device)
        case _utils.EyeTracker.SMI_ETG:
            from .SMI_ETG import getRecordingInfo
            rec_info = getRecordingInfo(source_dir)
        case _utils.EyeTracker.SeeTrue:
            from .SeeTrue import getRecordingInfo
            rec_info = getRecordingInfo(source_dir)
        case _utils.EyeTracker.Tobii_Glasses_2:
            from .tobii_G2 import getRecordingInfo
            rec_info = getRecordingInfo(source_dir)
        case _utils.EyeTracker.Tobii_Glasses_3:
            from .tobii_G3 import getRecordingInfo
            rec_info = getRecordingInfo(source_dir)

    if rec_info is not None and not isinstance(rec_info,list):
        rec_info = [rec_info]
    return rec_info


# single front end to the various device import functions. Step 1 of our 3-step process
def do_import(output_dir: str | pathlib.Path, source_dir: str | pathlib.Path = None, device: str | _utils.EyeTracker = None, rec_info: _utils.Recording = None):
    output_dir = pathlib.Path(output_dir)

    if rec_info is not None:
        if isinstance(rec_info,list):
            raise ValueError('You should provide a single Recording to this function''s "rec_info" input argument, not a list of Recordings.')

    if source_dir is None:
        if rec_info is None:
            raise RuntimeError('Either the "source_dir" or the "rec_info" input argument should be set.')
        else:
            source_dir  = pathlib.Path(rec_info.source_directory)
    else:
        source_dir  = pathlib.Path(source_dir)
        if rec_info is not None:
            if pathlib.Path(rec_info.source_directory) != source_dir:
                raise ValueError(f"The provided source_dir ({source_dir}) does not equal the source directory set in rec_info ({rec_info.source_directory}).")

    if device is None and rec_info is None:
        raise RuntimeError('Either the "device" or the "rec_info" input argument should be set.')
    if device is not None:
        device = _utils.type_string_to_enum(device)
    if rec_info is not None:
        if device is not None:
            if rec_info.eye_tracker != device:
                raise ValueError(f'Provided device ({device.value}) did not match device specific in rec_info ({rec_info.eye_tracker.value}). Provide matching values or do not provide the device input argument.')
        else:
            device = rec_info.eye_tracker

    # check output directory, if possible
    if rec_info is not None:
        if not rec_info.proc_directory_name:
            rec_info.proc_directory_name = _utils.make_fs_dirname(rec_info, output_dir)
        new_dir = output_dir / rec_info.proc_directory_name
        if new_dir.is_dir():
            raise RuntimeError(f'Output directory specified in rec_info ({rec_info.proc_directory_name}) already exists in the output_dir ({output_dir}). Cannot use.')

    # do the actual import/pre-process
    match device:
        case _utils.EyeTracker.Pupil_Core:
            rec_info = pupil_core(output_dir, source_dir, rec_info)
        case _utils.EyeTracker.Pupil_Invisible:
            rec_info = pupil_invisible(output_dir, source_dir, rec_info)
        case _utils.EyeTracker.SMI_ETG:
            rec_info = SMI_ETG(output_dir, source_dir, rec_info)
        case _utils.EyeTracker.SeeTrue:
            rec_info = SeeTrue(output_dir, source_dir, rec_info)
        case _utils.EyeTracker.Tobii_Glasses_2:
            rec_info = tobii_G2(output_dir, source_dir, rec_info)
        case _utils.EyeTracker.Tobii_Glasses_3:
            rec_info = tobii_G3(output_dir, source_dir, rec_info)
            
    if rec_info is not None and not isinstance(rec_info,list):
        rec_info = [rec_info]
    return rec_info


_status_file = 'glassesValidator.recording'
def _create_recording_status_file(file: pathlib.Path):
    task_status_dict = {str(getattr(_utils.Task,x)): _utils.Status.Not_Started for x in _utils.Task.__members__ if x not in ['Not_Imported', 'Unknown']}

    with open(file, 'w') as f:
        json.dump(task_status_dict, f, cls=_utils.CustomTypeEncoder)


def get_recording_status(path: str | pathlib.Path, create_if_missing = False):
    path = pathlib.Path(path)

    file = path / _status_file
    if not file.is_file():
        _create_recording_status_file(file)

    with open(file, 'r') as f:
        return json.load(f, object_hook=_utils.json_reconstitute)


def update_recording_status(path: str | pathlib.Path, task: _utils.Task, status: _utils.Status):
    rec_status = get_recording_status(path)

    rec_status[str(task)] = status

    file = path / _status_file
    with open(file, 'w') as f:
        json.dump(rec_status, f, cls=_utils.CustomTypeEncoder)

    return rec_status