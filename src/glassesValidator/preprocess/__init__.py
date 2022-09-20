# -*- coding: utf-8 -*-
import pathlib
from .. import utils as _utils

from .tobii_G2 import preprocessData as tobii_G2
from .tobii_G3 import preprocessData as tobii_G3
from .SMI_ETG import preprocessData as SMI_ETG
from .SeeTrue import preprocessData as SeeTrue

def pupil_core(source_dir: str | pathlib.Path, output_dir: str | pathlib.Path):
    from .pupilLabs import preprocessData
    preprocessData(source_dir, output_dir,'Pupil Core')


def pupil_invisible(source_dir: str | pathlib.Path, output_dir: str | pathlib.Path):
    from .pupilLabs import preprocessData
    preprocessData(source_dir, output_dir,'Pupil Invisible')

    
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