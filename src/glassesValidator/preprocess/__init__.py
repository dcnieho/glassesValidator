# -*- coding: utf-8 -*-
import pathlib

from .tobii_G2 import preprocessData as tobii_G2
from .tobii_G3 import preprocessData as tobii_G3
from .SMI_ETG import preprocessData as SMI_ETG
from .SeeTrue import preprocessData as SeeTrue

def pupil_core(inputDir, outputDir):
    from .pupilLabs import preprocessData
    preprocessData(inputDir, outputDir,'Pupil Core')


def pupil_invisible(inputDir, outputDir):
    from .pupilLabs import preprocessData
    preprocessData(inputDir, outputDir,'Pupil Invisible')
    
from .. import utils as _utils
def getRecordingInfo(inputDir: str | pathlib.Path, device: str | _utils.Type):
    inputDir  = pathlib.Path(inputDir)
    device = _utils.type_string_to_enum(device)

    file = inputDir / 'recording_glassesValidator.json'
    if file.is_file():
        return [_utils.Recording.load_from_json(file)]

    recInfo = None
    match device:
        case _utils.Type.Pupil_Core:
            from .pupilLabs import getRecordingInfo
            recInfo = getRecordingInfo(inputDir, device)
        case _utils.Type.Pupil_Invisible:
            from .pupilLabs import getRecordingInfo
            recInfo = getRecordingInfo(inputDir, device)
        case _utils.Type.SMI_ETG:
            from .SMI_ETG import getRecordingInfo
            recInfo = getRecordingInfo(inputDir)
        case _utils.Type.SeeTrue:
            from .SeeTrue import getRecordingInfo
            recInfo = getRecordingInfo(inputDir)
        case _utils.Type.Tobii_Glasses_2:
            from .tobii_G2 import getRecordingInfo
            recInfo = getRecordingInfo(inputDir)
        case _utils.Type.Tobii_Glasses_3:
            from .tobii_G3 import getRecordingInfo
            recInfo = getRecordingInfo(inputDir)

    if recInfo is not None and not isinstance(recInfo,list):
        recInfo = [recInfo]
    return recInfo