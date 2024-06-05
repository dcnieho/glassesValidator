# -*- coding: utf-8 -*-
import pathlib
import pathvalidate

from glassesTools.recording  import Recording
from glassesTools.eyetracker import EyeTracker
from glassesTools.importing  import check_device, do_import as _do_impt, get_recording_info

from .. import utils

# single front end to the various device import functions. Step 1 of our 3-step process
def do_import(output_dir: str | pathlib.Path = None, source_dir: str | pathlib.Path = None, device: str | EyeTracker = None, rec_info: Recording = None, copy_scene_video = True) -> Recording:
    # output_dir is the folder in which the working directory will be created
    if rec_info is not None:
        if isinstance(rec_info,list):
            raise ValueError('You should provide a single Recording to this function''s "rec_info" input argument, not a list of Recordings.')

    device, rec_info = check_device(device, rec_info)
    # ensure there is an output directory
    if rec_info is not None:
        if not rec_info.working_directory:
            rec_info.working_directory = output_dir / make_fs_dirname(rec_info, output_dir)

    # do the import
    rec_info = _do_impt(None, source_dir, device, rec_info, copy_scene_video)

    # make sure there is a processing status file
    utils.get_recording_status(rec_info.working_directory, create_if_missing=True)
    # write status file to indicate import finished
    utils.update_recording_status(rec_info.working_directory, utils.Task.Imported, utils.Status.Finished)

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


__all__ = ['get_recording_info','do_import']