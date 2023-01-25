import dataclasses
from enum import Enum, auto
import pathlib

from ...utils import AutoName, Recording, Task


class CounterContext:
    _count = -1     # so that first number is 0

    def __enter__(self):
        self._count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aenter__(self):
        self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

    def get_count(self):
        return self._count

    def set_count(self, count):
        self._count = count


class DefaultStyleDark:
    accent        = "#d46220"
    alt_bg        = "#181818"
    bg            = "#0a0a0a"
    border        = "#454545"
    corner_radius = 6
    text          = "#ffffff"
    text_dim      = "#808080"


class DefaultStyleLight:
    accent        = "#d46220"
    alt_bg        = "#e1e1e1"
    bg            = "#f0f0f0"
    border        = "#999999"
    corner_radius = 6
    text          = "#000000"
    text_dim      = "#999999"


@dataclasses.dataclass
class SortSpec:
    index: int
    reverse: bool


class FilterMode(AutoName):
    Choose      = auto()
    Eye_Tracker = auto()
    Task_State  = auto()
filter_mode_names = [x.value for x in FilterMode]

# summary version of task state, for client presentation
class TaskSimplified(AutoName):
    Not_Imported    = auto()
    Imported        = auto()
    Coded           = auto()
    Processed       = auto()
    Unknown         = auto()
simplified_task_names = [x.value for x in TaskSimplified]

def get_simplified_task_state(task: Task):
    match task:
        # before stage 1
        case Task.Not_Imported:
            return TaskSimplified.Not_Imported
        # after stage 1
        case Task.Imported:
            return TaskSimplified.Imported
        # after stage 2 / during stage 3
        case Task.Coded | Task.Markers_Detected | Task.Gaze_Tranformed_To_Poster | Task.Target_Offsets_Computed | Task.Fixation_Intervals_Determined:
            return TaskSimplified.Coded
        # after stage 3:
        case Task.Data_Quality_Calculated:
            return TaskSimplified.Processed
        # other
        case _: # includes Task.Unknown, Task.Make_Video
            return TaskSimplified.Unknown

@dataclasses.dataclass
class Filter:
    mode: FilterMode
    invert = False
    match = None

    def __post_init__(self):
        self.id = id(self)


class Os(Enum):
    Windows = auto()
    MacOS   = auto()
    Linux   = auto()


class MsgBox(Enum):
    question= auto()
    info    = auto()
    warn    = auto()
    error   = auto()


@dataclasses.dataclass
class Settings:
    config_dir                  : str
    confirm_on_remove           : bool
    continue_process_after_code : bool
    dq_use_viewpos_vidpos_homography : bool
    dq_use_pose_vidpos_homography : bool
    dq_use_pose_vidpos_ray      : bool
    dq_use_pose_left_eye        : bool
    dq_use_pose_right_eye       : bool
    dq_use_pose_left_right_avg  : bool
    dq_report_data_loss         : bool
    process_workers             : int
    render_when_unfocused       : bool
    show_advanced_options       : bool
    show_remove_btn             : bool
    style_accent                : tuple[float]
    style_alt_bg                : tuple[float]
    style_bg                    : tuple[float]
    style_border                : tuple[float]
    style_color_recording_name  : bool
    style_corner_radius         : int
    style_text                  : tuple[float]
    style_text_dim              : tuple[float]
    vsync_ratio                 : int

@dataclasses.dataclass
class JobDescription:
    id:                 int
    payload:            Recording
    project_path:       pathlib.Path
    task:               Task
    should_chain_next:  bool

class ProcessState(Enum):
    Pending     = auto()
    Running     = auto()
    Canceled    = auto()
    Failed      = auto()
    Completed   = auto()