import subprocess
import pathlib
import asyncio
import natsort
import os
import shutil
import pandas as pd


from .structs import JobDescription, MsgBox, Os
from . import globals, async_thread, db, gui, msgbox, process_pool, utils
from ...utils import EyeTracker, Recording, Task, eye_tracker_names, get_next_task, make_fs_dirname
from ... import config, preprocess, process, utils as gv_utils
from ...process import DataQualityType, _collect_data_quality, _summarize_and_store_data_quality



def open_url(path: str):
    # this works for files, folders and URLs
    if globals.os is Os.Windows:
        os.startfile(path)
    else:
        if globals.os is Os.Linux:
            open_util = "xdg-open"
        elif globals.os is Os.MacOS:
            open_util = "open"
        async_thread.run(asyncio.create_subprocess_exec(
            open_util, path,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ))

def open_folder(path: pathlib.Path):
    if not path.is_dir():
        utils.push_popup(msgbox.msgbox, "Folder not found", f"The folder you're trying to open\n{path}\ncould not be found.", MsgBox.warn)
        return
    open_url(str(path))


async def _deploy_config(conf_dir: pathlib.Path):
    config.deploy_validation_config(conf_dir)

async def deploy_config(project_path: str|pathlib.Path, config_dir: str):
    if not config_dir:
        utils.push_popup(msgbox.msgbox, "Cannot deploy", "Configuration directory name cannot be an empty value", MsgBox.error)
        return

    conf_dir = pathlib.Path(project_path) / config_dir
    if not conf_dir.is_dir():
        conf_dir.mkdir()
        await _deploy_config(conf_dir)
    else:
        buttons = {
            "󰄬 Yes": lambda: async_thread.run(_deploy_config(conf_dir)),
            "󰜺 No": None
        }
        utils.push_popup(msgbox.msgbox, "Deploy configuration", f"The folder {conf_dir} already exist. Do you want to deploy a configuration to this folder,\npotentially overwriting any configuration that is already there?", MsgBox.warn, buttons)

async def deploy_poster_pdf(dir: str|pathlib.Path):
    config.poster.deploy_default_pdf(dir)

async def remove_recording_working_dir(rec: Recording, project_path: pathlib.Path = None):
    if rec.proc_directory_name:
        if project_path:
            rec_path =         project_path / rec.proc_directory_name
        else:
            rec_path = globals.project_path / rec.proc_directory_name
        if rec_path.is_dir():
            shutil.rmtree(rec_path)

        # also set recording state back to not imported
        # NB: this might get called from remove_recording.remove_callback() below
        # after the recording is already removed from the database. That is not
        # an issue because db.update_recording() will be effectively no-op
        rec.task = Task.Not_Imported
        async_thread.run(db.update_recording(rec, "task"))


async def remove_recording(rec: Recording, bypass_confirm=False):
    def remove_callback():
        if rec.id in globals.jobs:
            process_pool.cancel_job(globals.jobs[rec.id].id)
        del globals.recordings[rec.id]
        del globals.selected_recordings[rec.id]
        async_thread.run(db.remove_recording(rec.id))

        if rec.proc_directory_name:
            async_thread.run(remove_recording_working_dir(rec))

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
            recordings[id].proc_directory_name = make_fs_dirname(recordings[id], globals.project_path)
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

    recording_list = gui.RecordingTable(recordings_to_add, recordings_selected_to_add, is_adder_popup=True)
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

async def process_recording(rec: Recording, task: Task=None, chain=True):
    # find what is the next task to do for this recording
    if task is None:
        task = get_next_task(rec.task)

    # get function for task
    working_dir = globals.project_path / rec.proc_directory_name
    kwargs = {}
    match task:
        case Task.Imported:
            fun = preprocess.do_import
            args = (globals.project_path,)
            kwargs['rec_info'] = rec
        case Task.Coded | Task.Markers_Detected | Task.Gaze_Tranformed_To_Poster | Task.Target_Offsets_Computed | Task.Fixation_Intervals_Determined | Task.Data_Quality_Calculated | Task.Make_Video:
            match task:
                case Task.Coded:
                    fun = process.code_marker_interval
                case Task.Markers_Detected:
                    fun = process.detect_markers
                case Task.Gaze_Tranformed_To_Poster:
                    fun = process.gaze_to_poster
                case Task.Target_Offsets_Computed:
                    fun = process.compute_offsets_to_targets
                case Task.Fixation_Intervals_Determined:
                    fun = process.determine_fixation_intervals
                    kwargs['do_global_shift'] = globals.settings.fix_assign_do_global_shift
                    kwargs['max_dist_fac']    = globals.settings.fix_assign_max_dist_fac
                case Task.Data_Quality_Calculated:
                    fun = process.calculate_data_quality
                    kwargs['allow_dq_fallback'] = True
                    kwargs['dq_types'] = []
                    if globals.settings.dq_use_viewpos_vidpos_homography:
                        kwargs['dq_types'].append(DataQualityType.viewpos_vidpos_homography)
                    if globals.settings.dq_use_pose_vidpos_homography:
                        kwargs['dq_types'].append(DataQualityType.pose_vidpos_homography)
                    if globals.settings.dq_use_pose_vidpos_ray:
                        kwargs['dq_types'].append(DataQualityType.pose_vidpos_ray)
                    if globals.settings.dq_use_pose_world_eye:
                        kwargs['dq_types'].append(DataQualityType.pose_world_eye)
                    if globals.settings.dq_use_pose_left_eye:
                        kwargs['dq_types'].append(DataQualityType.pose_left_eye)
                    if globals.settings.dq_use_pose_right_eye:
                        kwargs['dq_types'].append(DataQualityType.pose_right_eye)
                    if globals.settings.dq_use_pose_left_right_avg:
                        kwargs['dq_types'].append(DataQualityType.pose_left_right_avg)
                    kwargs['include_data_loss'] = globals.settings.dq_report_data_loss
                case Task.Make_Video:
                    fun = gv_utils.make_video
            args = (working_dir,)
            if globals.settings.config_dir and (config_dir := globals.project_path / globals.settings.config_dir).is_dir() and task!=Task.Data_Quality_Calculated:
                kwargs['config_dir'] = config_dir

        # other, includes Task.Unknown and None (occurs when all already done), nothing to do if no specific task specified:
        case _:
            fun = None  # nothing to do

    # exit if nothing to do
    if fun is None:
        return

    # special case if its a marker coding task, of which we can have only one at a time. If we already have a marker coding task
    # store this task in a separate task queue instead of launching it now
    should_launch_task = task!=Task.Coded or not any((globals.jobs[j].task==Task.Coded for j in globals.jobs))

    job = JobDescription(None, rec, globals.project_path, task, chain)
    if should_launch_task:
        # launch task
        job_id = process_pool.run(fun,*args,**kwargs)

        # store to job queue
        job.id = job_id
        globals.jobs[rec.id] = job
    else:
        globals.coding_job_queue[rec.id] = job

async def process_recordings(ids: list[int], task: Task=None, chain=True):
    for rec_id in ids:
        await process_recording(globals.recordings[rec_id], task, chain)

async def cancel_processing_recordings(ids: list[int]):
    for rec_id in ids:
        if rec_id in globals.jobs:
            process_pool.cancel_job(globals.jobs[rec_id].id)
        if rec_id in globals.coding_job_queue:
            del globals.coding_job_queue[rec_id]

async def export_data_quality(ids: list[int]):
    # 1. collect all data quality from the selected recordings
    rec_dirs = [globals.project_path / globals.recordings[id].proc_directory_name for id in ids]
    df, default_dq_type, targets = _collect_data_quality(rec_dirs)
    if df is None:
        utils.push_popup(msgbox.msgbox, "Export error", "There is no data quality for the selected recordings. Did you code any validation intervals (see manual)?", MsgBox.error)
        return

    # 2. prep popup
    pop_data = {}

    # data quality type
    typeIdx = df.index.names.index('type')
    pop_data['dq_types'] = sorted(list(df.index.levels[typeIdx]), key=lambda dq: dq.value)
    pop_data['dq_types_sel'] = [False for i in pop_data['dq_types']]
    if globals.settings.dq_use_viewpos_vidpos_homography and DataQualityType.viewpos_vidpos_homography in pop_data['dq_types']:
        pop_data['dq_types_sel'][pop_data['dq_types'].index(DataQualityType.viewpos_vidpos_homography)] = True
    if globals.settings.dq_use_pose_vidpos_homography and DataQualityType.pose_vidpos_homography in pop_data['dq_types']:
        pop_data['dq_types_sel'][pop_data['dq_types'].index(DataQualityType.pose_vidpos_homography)] = True
    if globals.settings.dq_use_pose_vidpos_ray and DataQualityType.pose_vidpos_ray in pop_data['dq_types']:
        pop_data['dq_types_sel'][pop_data['dq_types'].index(DataQualityType.pose_vidpos_ray)] = True
    if globals.settings.dq_use_pose_left_eye and DataQualityType.pose_left_eye in pop_data['dq_types']:
        pop_data['dq_types_sel'][pop_data['dq_types'].index(DataQualityType.pose_left_eye)] = True
    if globals.settings.dq_use_pose_right_eye and DataQualityType.pose_right_eye in pop_data['dq_types']:
        pop_data['dq_types_sel'][pop_data['dq_types'].index(DataQualityType.pose_right_eye)] = True
    if globals.settings.dq_use_pose_left_right_avg and DataQualityType.pose_left_right_avg in pop_data['dq_types']:
        pop_data['dq_types_sel'][pop_data['dq_types'].index(DataQualityType.pose_left_right_avg)] = True

    if not any(pop_data['dq_types_sel']):
        pop_data['dq_types_sel'][pop_data['dq_types'].index(default_dq_type)] = True

    # targets
    pop_data['targets']     = targets
    pop_data['targets_sel'] = [True for i in pop_data['targets']]
    pop_data['targets_avg'] = False

    # other settings
    pop_data['include_data_loss'] = globals.settings.dq_report_data_loss

    # 3. show popup
    def show_config_popup():
        nonlocal pop_data
        globals.gui.draw_dq_export_config_popup(pop_data)

    buttons = {
        "󰄬 Continue": lambda: async_thread.run(_export_data_quality(df,pop_data)),
        "󰜺 Cancel": None
    }
    utils.push_popup(lambda: utils.popup("Data Quality Export", show_config_popup, buttons = buttons, closable=True, outside=False))

async def _export_data_quality(df: pd.DataFrame, pop_data: dict):
    dq_types = [dq for i,dq in enumerate(pop_data['dq_types']) if pop_data['dq_types_sel'][i]]
    targets  = [t for i,t in enumerate(pop_data['targets']) if pop_data['targets_sel'][i]]
    _summarize_and_store_data_quality(df, globals.project_path, dq_types, targets, pop_data['targets_avg'], pop_data['include_data_loss'])