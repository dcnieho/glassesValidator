import pathlib
import enum
import json
import csv

from .makeVideo import process as make_video

from glassesTools import utils
from glassesTools.recording import Recording
from glassesTools.eyetracker import EyeTracker

# this is a bit of a mix of a list of the various tasks, and a status-keeper so we know where we are in the process.
# hence the Not_imported and Unknown states are mixed in, and all names are past tense verbs
# To get actual task versions, use task_names_friendly
class Task(utils.AutoName):
    Not_Imported                    = enum.auto()
    Imported                        = enum.auto()
    Coded                           = enum.auto()
    Markers_Detected                = enum.auto()
    Gaze_Tranformed_To_Poster       = enum.auto()
    Target_Offsets_Computed         = enum.auto()
    Fixation_Intervals_Determined   = enum.auto()
    Data_Quality_Calculated         = enum.auto()
    # special task that is separate from status
    Make_Video                      = enum.auto()
    Unknown                         = enum.auto()
task_names = [x.value for x in Task]
utils.register_type(utils.CustomTypeEntry(Task,'__enum.Task__', utils.enum_val_2_str, lambda x: getattr(Task, x.split('.')[1])))

def get_task_name_friendly(name: str | Task):
    if isinstance(name,Task):
        name = name.name

    match name:
        case 'Imported':
            return 'Import'
        case 'Coded':
            return 'Code Intervals'
        case 'Markers_Detected':
            return 'Detect Markers'
        case 'Gaze_Tranformed_To_Poster':
            return 'Tranform Gaze To Poster'
        case 'Target_Offsets_Computed':
            return 'Compute Target Offsets'
        case 'Fixation_Intervals_Determined':
            return 'Determine Fixation Intervals'
        case 'Data_Quality_Calculated':
            return 'Calculate Data Quality'
        case 'Make_Video':
            return 'Make Video'
    return '' # 'Not_Imported', 'Unknown'

task_names_friendly = [get_task_name_friendly(x) for x in Task]   # non verb version

def get_next_task(task: Task) -> Task:
    match task:
        # stage 1
        case Task.Not_Imported:
            next_task = Task.Imported

        # stage 2
        case Task.Imported:
            next_task = Task.Coded

        # stage 3 substeps
        case Task.Coded:
            next_task = Task.Markers_Detected
        case Task.Markers_Detected:
            next_task = Task.Gaze_Tranformed_To_Poster
        case Task.Gaze_Tranformed_To_Poster:
            next_task = Task.Target_Offsets_Computed
        case Task.Target_Offsets_Computed:
            next_task = Task.Fixation_Intervals_Determined
        case Task.Fixation_Intervals_Determined:
            next_task = Task.Data_Quality_Calculated

        # other, includes Task.Data_Quality_Calculated (all already done), nothing to do if no specific task specified:
        case _:
            next_task = None
    return next_task

class Status(utils.AutoName):
    Not_Started     = enum.auto()
    Running         = enum.auto()
    Finished        = enum.auto()
    Errored         = enum.auto()
status_names = [x.value for x in Status]
utils.register_type(utils.CustomTypeEntry(Status,'__enum.Status__', utils.enum_val_2_str, lambda x: getattr(Status, x.split('.')[1])))


_status_file = 'glassesValidator.recording'
def _create_recording_status_file(file: pathlib.Path):
    task_status_dict = {utils.enum_val_2_str(getattr(Task,x)): Status.Not_Started for x in Task.__members__ if x not in ['Not_Imported', 'Make_Video', 'Unknown']}

    with open(file, 'w') as f:
        json.dump(task_status_dict, f, cls=utils.CustomTypeEncoder)


def get_recording_status(path: str | pathlib.Path, create_if_missing = False, skip_if_missing=False):
    path = pathlib.Path(path)

    file = path / _status_file
    if not file.is_file():
        if create_if_missing:
            _create_recording_status_file(file)
        elif skip_if_missing:
            return None

    with open(file, 'r') as f:
        return json.load(f, object_hook=utils.json_reconstitute)

def get_last_finished_step(status: dict[str,Status]):
    last = Task.Not_Imported
    while (next_task:=get_next_task(last)) is not None:
        if status[utils.enum_val_2_str(next_task)] != Status.Finished:
            break
        last = next_task

    return last

def update_recording_status(path: str | pathlib.Path, task: Task, status: Status, skip_if_missing=False):
    rec_status = get_recording_status(path, skip_if_missing=skip_if_missing)
    if rec_status is None and skip_if_missing:
        return None

    # set status of indicated task
    rec_status[utils.enum_val_2_str(task)] = status
    # set all later tasks to not started as they would have to be rerun when an earlier tasks is rerun
    next_task = task
    while (next_task:=get_next_task(next_task)) is not None:
        rec_status[utils.enum_val_2_str(next_task)] = Status.Not_Started

    file = path / _status_file
    with open(file, 'w') as f:
        json.dump(rec_status, f, cls=utils.CustomTypeEncoder, indent=2)

    return rec_status


def readMarkerIntervalsFile(fileName) -> list[list[int]]:
    analyzeFrames = []
    if pathlib.Path(fileName).is_file():
        with open(fileName, 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                analyzeFrames.append([int(float(entry['start_frame'])), int(float(entry['end_frame']))])

    return None if len(analyzeFrames)==0 else analyzeFrames

def writeMarkerIntervalsFile(fileName, intervals: list[int]|list[list[int]]):
    if intervals and isinstance(intervals[0],list):
        # flatten
        intervals = [i for x in intervals for i in x]
    with open(fileName, 'w', newline='') as file:
        csv_writer = csv.writer(file, delimiter='\t')
        csv_writer.writerow(['start_frame', 'end_frame'])
        for f in range(0,len(intervals)-1,2):   # -1 to make sure we don't write out incomplete intervals
            csv_writer.writerow(intervals[f:f+2])



__all__ = ['make_video','Recording','EyeTracker']