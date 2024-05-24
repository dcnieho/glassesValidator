# -*- coding: utf-8 -*-
import pathlib
import pathvalidate

from glassesTools.recording  import Recording
from glassesTools.eyetracker import EyeTracker

from glassesTools.importing import adhawk_mindlink, pupil_core, pupil_invisible, pupil_neon, SeeTrue_STONE, SMI_ETG, tobii_G2, tobii_G3
from glassesTools.importing import check_device, check_source_dir, check_output_dir, get_recording_info

from .. import utils

# single front end to the various device import functions. Step 1 of our 3-step process
def do_import(output_dir: str | pathlib.Path = None, source_dir: str | pathlib.Path = None, device: str | EyeTracker = None, rec_info: Recording = None) -> Recording:
    if rec_info is not None:
        if isinstance(rec_info,list):
            raise ValueError('You should provide a single Recording to this function''s "rec_info" input argument, not a list of Recordings.')

    device, rec_info = check_device(device, rec_info)
    source_dir, rec_info = check_source_dir(source_dir, rec_info)

    # ensure there is an output directory
    if rec_info is not None:
        if not rec_info.working_directory:
            rec_info.working_directory = output_dir / make_fs_dirname(rec_info, output_dir)
    output_dir, rec_info = check_output_dir(None, rec_info)

    # do the actual import/pre-process
    match device:
        case EyeTracker.AdHawk_MindLink:
            rec_info = adhawk_mindlink(output_dir, source_dir, rec_info)
        case EyeTracker.Pupil_Core:
            rec_info = pupil_core(output_dir, source_dir, rec_info)
        case EyeTracker.Pupil_Invisible:
            rec_info = pupil_invisible(output_dir, source_dir, rec_info)
        case EyeTracker.Pupil_Neon:
            rec_info = pupil_neon(output_dir, source_dir, rec_info)
        case EyeTracker.SeeTrue_STONE:
            rec_info = SeeTrue_STONE(output_dir, source_dir, rec_info)
        case EyeTracker.SMI_ETG:
            rec_info = SMI_ETG(output_dir, source_dir, rec_info)
        case EyeTracker.Tobii_Glasses_2:
            rec_info = tobii_G2(output_dir, source_dir, rec_info)
        case EyeTracker.Tobii_Glasses_3:
            rec_info = tobii_G3(output_dir, source_dir, rec_info)

    # make sure there is a processing status file
    utils.get_recording_status(output_dir, create_if_missing=True)
    # write status file to indicate import finished
    utils.update_recording_status(output_dir, utils.Task.Imported, utils.Status.Finished)

    return rec_info


# function for making working folder names
def make_fs_dirname(rec_info: Recording, output_dir: pathlib.Path = None):
    if rec_info.participant:
        dirname = f"{rec_info.eye_tracker.value}_{rec_info.participant}_{rec_info.name}"
    else:
        dirname = f"{rec_info.eye_tracker.value}_{rec_info.name}"

    # make sure its a valid path
    dirname = pathvalidate.sanitize_filename(dirname)

    # check it doesn't already exist
    if output_dir is not None:
        if (output_dir / dirname).is_dir():
            # add _1, _2, etc, until we find a unique name
            fver = 1
            while (output_dir / f'{dirname}_{fver}').is_dir():
                fver += 1

            dirname = f'{dirname}_{fver}'

    return dirname


__all__ = ['adhawk_mindlink','pupil_core','pupil_invisible','pupil_neon','SeeTrue_STONE','SMI_ETG','tobii_G2','tobii_G3',
           'get_recording_info','do_import']