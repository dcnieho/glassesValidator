import subprocess
import pathlib
import asyncio
import natsort
import os
import shutil


from .structs import JobDescription, MsgBox, Os
from . import globals, async_thread, db, gui, msgbox, process_pool, utils
from ...utils import EyeTracker, Recording, Task, eye_tracker_names
from ... import preprocess
from ... import process




def open_folder(path: pathlib.Path):
    if not path.is_dir():
        utils.push_popup(msgbox.msgbox, "Folder not found", f"The folder you're trying to open\n{path}\ncould not be found.", MsgBox.warn)
        return
    if globals.os is Os.Windows:
        os.startfile(str(path))
    else:
        if globals.os is Os.Linux:
            open_util = "xdg-open"
        elif globals.os is Os.MacOS:
            open_util = "open"
        async_thread.run(asyncio.create_subprocess_exec(
            open_util, str(path),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ))


def remove_recording(rec: Recording, bypass_confirm=False):
    def remove_callback():
        del globals.recordings[rec.id]
        del globals.selected_recordings[rec.id]
        async_thread.run(db.remove_recording(rec.id))

        if rec.proc_directory_name:
            rec_path = globals.project_path / rec.proc_directory_name
            if rec_path.is_dir():
                async_thread.run(lambda: shutil.rmtree(rec_path))

    if not bypass_confirm and globals.settings.confirm_on_remove:
        buttons = {
            "󰄬 Yes": remove_callback,
            "󰜺 No": None
        }
        utils.push_popup(msgbox.msgbox, "Remove recording", f"Are you sure you want to remove {rec.name} from your list?", MsgBox.warn, buttons)
    else:
        remove_callback()


async def _add_recordings(recordings: dict[int, Recording], selected: dict[int, bool]):
    for id in recordings:
        if selected[id]:
            recordings[id].task = Task.Not_Imported
            rid = await db.add_recording(recordings[id])
            await db.load_recordings(rid)
        
async def _show_addable_recordings(paths: list[pathlib.Path], eye_tracker: EyeTracker):
    # notify we're preparing the recordings to be opened
    def prepping_recs_popup():
        globals.gui.draw_preparing_recordings_for_import_popup(eye_tracker)
    utils.push_popup(lambda: utils.popup("Preparing import", prepping_recs_popup, buttons = None, closable=False, outside=False))

    # step 1, find what recordings of this type of eye tracker are in the path
    all_recs = []
    dup_recs = []
    for p in paths:
        all_dirs = utils.fast_scandir(p)
        all_dirs.append(p)
        for d in all_dirs:
            # check if dir is a valid recording
            if (recs:=preprocess.get_recording_info(d, eye_tracker)) is not None:
                for rec in recs:
                    # skip duplicates
                    if rec.source_directory not in (r.source_directory for r in globals.recordings.values()):
                        all_recs.append(rec)
                    else:
                        dup_recs.append(rec)

    # sort in order natural for OS
    all_recs = natsort.os_sorted(all_recs, lambda rec: rec.source_directory)
    
    # get ready to show result
    # 1. remove progress popup
    del globals.popup_stack[-1]

    # 2. if nothing importable found, notify
    if not all_recs:
        if dup_recs:
            dup_recs = natsort.os_sorted(dup_recs, lambda rec: rec.source_directory)
            msg = f"{eye_tracker.value} recordings were found in the specified import paths, but could not be imported as they are already part of this glassesValidator project."
            more="Duplicates that were not imported:\n"+('\n'.join([str(r.source_directory) for r in dup_recs]))
        else:
            msg = f"No {eye_tracker.value} recordings were found among the specified import paths."
            more = None

        utils.push_popup(msgbox.msgbox, "Nothing to import", msg, MsgBox.warn, more=more)
        return

    # 3. if something importable found, show to user so they can select the ones they want
    # put in dict
    recordings_to_add = {}
    recordings_selected_to_add = {}
    for id,rec in enumerate(all_recs):
        recordings_to_add[id] = rec
        recordings_selected_to_add[id] = True

    recording_list = gui.RecordingTable(recordings_to_add, recordings_selected_to_add, True)
    def list_recs_popup():
        nonlocal recording_list
        globals.gui.draw_select_recordings_to_import(recording_list)

    buttons = {
        "󰄬 Continue": lambda: async_thread.run(_add_recordings(recordings_to_add, recordings_selected_to_add)),
        "󰜺 Cancel": None
    }
    utils.push_popup(lambda: utils.popup("Select recordings", list_recs_popup, buttons = buttons, closable=True, outside=False))

def add_recordings(paths: list[pathlib.Path]):
    combo_value = 0
    eye_tracker = EyeTracker(eye_tracker_names[combo_value])

    def add_recs_popup():
        nonlocal combo_value, eye_tracker
        combo_value, eye_tracker = globals.gui.draw_select_eye_tracker_popup(combo_value, eye_tracker)
                
    buttons = {
        "󰄬 Continue": lambda: async_thread.run(_show_addable_recordings(paths, eye_tracker)),
        "󰜺 Cancel": None
    }

    # ask what type of eye tracker we should be looking for
    utils.push_popup(lambda: utils.popup("Select eye tracker", add_recs_popup, buttons = buttons, closable=True, outside=True))

def _process_recording(rec: Recording, task: Task = None, chain=True):
    # find what is the next task to do for this recording
    if task is None:
        match rec.task:
            # stage 1
            case Task.Not_Imported:
                task = Task.Imported

            # stage 2
            case Task.Imported:
                task = Task.Coded

            # stage 3 substeps
            case Task.Coded:
                task = Task.Markers_Detected
            case Task.Markers_Detected:
                task = Task.Gaze_Tranformed_To_World
            case Task.Gaze_Tranformed_To_World:
                task = Task.Target_Offsets_Computed
            case Task.Target_Offsets_Computed:
                task = Task.Fixation_Intervals_Determined
            case Task.Fixation_Intervals_Determined:
                task = Task.Data_Quality_Calculated

            # other, includes Task.Data_Quality_Calculated (all already done), nothing to do if no specific task specified:
            case _:
                task = Task.Unknown

    # get function for task
    match task:
        case Task.Imported:
            fun = preprocess.do_import
            args = (globals.project_path,)
            kwargs = {'rec_info': rec}
        case Task.Coded:
            fun = preprocess.do_import
        case Task.Markers_Detected:
            fun = preprocess.do_import
        case Task.Gaze_Tranformed_To_World:
            fun = preprocess.do_import
        case Task.Target_Offsets_Computed:
            fun = preprocess.do_import
        case Task.Fixation_Intervals_Determined:
            fun = preprocess.do_import
        case Task.Data_Quality_Calculated:
            fun = preprocess.do_import
         
        # other, includes Task.Unknown (all already done), nothing to do if no specific task specified:
        case _:
            fun = None  # nothing to do

    # exit if nothing to do
    if fun is None:
        return

    # launch task
    job_id = process_pool.run(fun,*args,**kwargs)

    # store to job queue
    globals.jobs[rec.id] = JobDescription(job_id, rec, task)

async def process_recordings(ids: list[int], task: Task = None, chain=True):
    for rec_id in ids:
        _process_recording(globals.recordings[rec_id], task, chain)