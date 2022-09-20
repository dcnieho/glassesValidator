import dataclasses
from enum import Enum, auto

from ...utils import AutoName


class CounterContext:
    count = 0

    def __enter__(self):
        self.count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aenter__(self):
        self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)

    def set_count(self, count):
        self.count = count

        
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
    Status      = auto()

filter_mode_names = [getattr(FilterMode,x).value for x in FilterMode.__members__]


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
    confirm_on_remove           : bool
    refresh_workers             : int
    render_when_unfocused       : bool
    scroll_amount               : float
    scroll_smooth               : bool
    scroll_smooth_speed         : float
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