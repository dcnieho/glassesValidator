import pathlib
import enum
import csv
import json as std_json

from .makeVideo import process as make_video

from glassesTools import json, utils
from glassesTools.recording import Recording
from glassesTools.eyetracker import EyeTracker


def _decode_legacy_enum(value):
    enum_type = value.get('__enum__', '')
    if enum_type.startswith('EyeTracker.'):
        member_name = enum_type.split('.', 1)[1]
        if member_name == 'SeeTrue':
            member_name = 'SeeTrue_STONE'
        return getattr(EyeTracker, member_name)
    if enum_type.startswith('Status.'):
        return Status[enum_type.split('.', 1)[1]]
    if enum_type.startswith('Task.'):
        return Task[enum_type.split('.', 1)[1]]
    return json.json_decoder(value)


def get_recording_info_file(path: str | pathlib.Path) -> pathlib.Path | None:
    path = pathlib.Path(path)
    if path.is_dir():
        for name in (Recording.default_json_file_name, 'recording_glassesValidator.json'):
            info_file = path / name
            if info_file.is_file():
                return info_file
        return None
    return path if path.is_file() else None


def load_recording_info(path: str | pathlib.Path) -> Recording:
    path = pathlib.Path(path)
    info_file = get_recording_info_file(path)
    if info_file is None:
        return Recording.load_from_json(path)
    if info_file.name != 'recording_glassesValidator.json':
        return Recording.load_from_json(info_file)

    with info_file.open(encoding='utf-8') as f:
        info = std_json.load(f, object_hook=_decode_legacy_enum)
    info.pop('proc_directory_name', None)
    info.pop('working_directory', None)
    eye_tracker = info['eye_tracker']
    if isinstance(eye_tracker, str):
        if eye_tracker == 'SeeTrue':
            eye_tracker = 'SeeTrue STONE'
        if hasattr(EyeTracker, eye_tracker):
            eye_tracker = getattr(EyeTracker, eye_tracker)
        else:
            eye_tracker = EyeTracker(eye_tracker)
    info['eye_tracker'] = eye_tracker
    return Recording(**info, working_directory=info_file.parent)


# this is a bit of a mix of a list of the various tasks, and a status-keeper so we know where we are in the process.
# hence the Not_imported and Unknown states are mixed in, and all names are past tense verbs
# To get actual task versions, use task_names_friendly
class Task(utils.AutoName):
    Not_Imported                    = enum.auto()
    Imported                        = enum.auto()
    Coded                           = enum.auto()
    Markers_Detected                = enum.auto()
    Gaze_Tranformed_To_Plane        = enum.auto()
    Fixations_Classified            = enum.auto()
    Fixation_Assigned               = enum.auto()
    Data_Quality_Calculated         = enum.auto()
    # special task that is separate from status
    Make_Video                      = enum.auto()
    Unknown                         = enum.auto()
task_names = [x.value for x in Task]
json.register_type(json.TypeEntry(Task,'__enum.Task__', utils.enum_val_2_str, lambda x: getattr(Task, x.split('.')[1])))

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
        case 'Gaze_Tranformed_To_Plane':
            return 'Tranform Gaze To Plane'
        case 'Fixations_Classified':
            return 'Classify Fixations'
        case 'Fixation_Assigned':
            return 'Assign Fixations to Targets'
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
            next_task = Task.Gaze_Tranformed_To_Plane
        case Task.Gaze_Tranformed_To_Plane:
            next_task = Task.Fixations_Classified
        case Task.Fixations_Classified:
            next_task = Task.Fixation_Assigned
        case Task.Fixation_Assigned:
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
json.register_type(json.TypeEntry(Status,'__enum.Status__', utils.enum_val_2_str, lambda x: getattr(Status, x.split('.')[1])))


_status_file = 'glassesValidator.recording'
_legacy_task_status_keys = {
    'Task.Gaze_Tranformed_To_Poster': ('Task.Gaze_Tranformed_To_Plane',),
    'Task.Fixation_Intervals_Determined': ('Task.Fixations_Classified', 'Task.Fixation_Assigned'),
    'Task.Target_Offsets_Computed': ('Task.Data_Quality_Calculated',),
}


def _create_recording_status_file(file: pathlib.Path):
    task_status_dict = {utils.enum_val_2_str(getattr(Task,x)): Status.Not_Started for x in Task.__members__ if x not in ['Not_Imported', 'Make_Video', 'Unknown']}
    json.dump(task_status_dict, file)


def get_recording_status(path: str | pathlib.Path, create_if_missing = False, skip_if_missing=False):
    path = pathlib.Path(path)

    file = path / _status_file
    if not file.is_file():
        if create_if_missing:
            _create_recording_status_file(file)
        elif skip_if_missing:
            return None

    with file.open(encoding='utf-8') as f:
        recording_states = std_json.load(f, object_hook=_decode_legacy_enum)
    recording_states = {
        key: value if isinstance(value, Status) else Status(value)
        for key, value in recording_states.items()
    }
    for legacy_key, current_keys in _legacy_task_status_keys.items():
        if legacy_key in recording_states:
            for current_key in current_keys:
                recording_states.setdefault(current_key, recording_states[legacy_key])

    task = Task.Not_Imported
    while (task := get_next_task(task)) is not None:
        recording_states.setdefault(utils.enum_val_2_str(task), Status.Not_Started)
    return recording_states

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

    json.dump(rec_status, path / _status_file)

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