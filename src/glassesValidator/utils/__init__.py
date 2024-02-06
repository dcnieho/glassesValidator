import math
import cv2
import numpy as np
import pandas as pd
import csv
import itertools
import warnings
import pathlib
import bisect
from matplotlib import colors
import tempfile
import dataclasses
from enum import Enum, auto
import datetime
import json
import pathvalidate
import typing

from .mp4analyser import iso
from .. import config as gv_config

from .makeVideo import process as make_video


class Timestamp:
    def __init__(self, unix_time: int | float, format="%Y-%m-%d %H:%M:%S"):
        self.format = format
        self.display = ""
        self.value = 0
        self.update(unix_time)

    def update(self, unix_time: int | float):
        self.value = int(unix_time)
        if self.value == 0:
            self.display = ""
        else:
            self.display = datetime.datetime.fromtimestamp(unix_time).strftime(self.format)


def hex_to_rgba_0_1(hex):
    r = int(hex[1:3], base=16) / 255
    g = int(hex[3:5], base=16) / 255
    b = int(hex[5:7], base=16) / 255
    if len(hex) > 7:
        a = int(hex[7:9], base=16) / 255
    else:
        a = 1.0
    return (r, g, b, a)


def rgba_0_1_to_hex(rgba):
    r = "%.2x" % int(rgba[0] * 255)
    g = "%.2x" % int(rgba[1] * 255)
    b = "%.2x" % int(rgba[2] * 255)
    if len(rgba) > 3:
        a = "%.2x" % int(rgba[3] * 255)
    else:
        a = "FF"
    return f"#{r}{g}{b}{a}"


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.strip("_").replace("__", "-").replace("_", " ")


class EyeTracker(AutoName):
    AdHawk_MindLink = auto()
    Pupil_Core      = auto()
    Pupil_Invisible = auto()
    Pupil_Neon      = auto()
    SMI_ETG         = auto()
    SeeTrue         = auto()
    Tobii_Glasses_2 = auto()
    Tobii_Glasses_3 = auto()
    Unknown         = auto()
eye_tracker_names = [x.value for x in EyeTracker if x!=EyeTracker.Unknown]

EyeTracker.AdHawk_MindLink.color = hex_to_rgba_0_1("#001D7A")
EyeTracker.Pupil_Core     .color = hex_to_rgba_0_1("#E6194B")
EyeTracker.Pupil_Invisible.color = hex_to_rgba_0_1("#3CB44B")
EyeTracker.Pupil_Neon     .color = hex_to_rgba_0_1("#C6B41E")
EyeTracker.SMI_ETG        .color = hex_to_rgba_0_1("#4363D8")
EyeTracker.SeeTrue        .color = hex_to_rgba_0_1("#911EB4")
EyeTracker.Tobii_Glasses_2.color = hex_to_rgba_0_1("#F58231")
EyeTracker.Tobii_Glasses_3.color = hex_to_rgba_0_1("#F032E6")
EyeTracker.Unknown        .color = hex_to_rgba_0_1("#393939")

def type_string_to_enum(device: str):
    if isinstance(device, EyeTracker):
        return device

    if isinstance(device, str):
        if hasattr(EyeTracker, device):
            return getattr(EyeTracker, device)
        elif device in [e.value for e in EyeTracker]:
            return EyeTracker(device)
        else:
            raise ValueError(f"The string '{device}' is not a known eye tracker type. Known types: {[e.value for e in EyeTracker]}")
    else:
        raise ValueError(f"The variable 'device' should be a string with one of the following values: {[e.value for e in EyeTracker]}")


# this is a bit of a mix of a list of the various tasks, and a status-keeper so we know where we are in the process.
# hence the Not_imported and Unknown states are mixed in, and all names are past tense verbs
# To get actual task versions, use task_names_friendly
class Task(AutoName):
    Not_Imported                    = auto()
    Imported                        = auto()
    Coded                           = auto()
    Markers_Detected                = auto()
    Gaze_Tranformed_To_Poster       = auto()
    Target_Offsets_Computed         = auto()
    Fixation_Intervals_Determined   = auto()
    Data_Quality_Calculated         = auto()
    # special task that is separate from status
    Make_Video                      = auto()
    Unknown                         = auto()

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

task_names = [x.value for x in Task]
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

class Status(AutoName):
    Not_Started     = auto()
    Running         = auto()
    Finished        = auto()
    Errored         = auto()
status_names = [x.value for x in Status]


_status_file = 'glassesValidator.recording'
def _create_recording_status_file(file: pathlib.Path):
    task_status_dict = {str(getattr(Task,x)): Status.Not_Started for x in Task.__members__ if x not in ['Not_Imported', 'Make_Video', 'Unknown']}

    with open(file, 'w') as f:
        json.dump(task_status_dict, f, cls=CustomTypeEncoder)


def get_recording_status(path: str | pathlib.Path, create_if_missing = False):
    path = pathlib.Path(path)

    file = path / _status_file
    if not file.is_file():
        _create_recording_status_file(file)

    with open(file, 'r') as f:
        return json.load(f, object_hook=json_reconstitute)

def get_last_finished_step(status: typing.Dict[str,Status]):
    last = Task.Not_Imported
    while (next_task:=get_next_task(last)) is not None:
        if status[str(next_task)] != Status.Finished:
            break
        last = next_task

    return last

def update_recording_status(path: str | pathlib.Path, task: Task, status: Status):
    rec_status = get_recording_status(path)

    # set status of indicated task
    rec_status[str(task)] = status
    # set all later tasks to not started as they would have to be rerun when an earlier tasks is rerun
    next_task = task
    while (next_task:=get_next_task(next_task)) is not None:
        rec_status[str(next_task)] = Status.Not_Started

    file = path / _status_file
    with open(file, 'w') as f:
        json.dump(rec_status, f, cls=CustomTypeEncoder)

    return rec_status


@dataclasses.dataclass
class Recording:
    default_json_file_name      : typing.ClassVar[str] = 'recording_glassesValidator.json'

    id                          : int           = None
    name                        : str           = ""
    source_directory            : pathlib.Path  = ""
    proc_directory_name         : str           = ""
    start_time                  : Timestamp     = 0
    duration                    : int           = None
    eye_tracker                 : EyeTracker    = EyeTracker.Unknown
    project                     : str           = ""
    participant                 : str           = ""
    firmware_version            : str           = ""
    glasses_serial              : str           = ""
    recording_unit_serial       : str           = ""
    recording_software_version  : str           = ""
    scene_camera_serial         : str           = ""
    task                        : Task          = Task.Unknown

    def store_as_json(self, path: str | pathlib.Path):
        path = pathlib.Path(path)
        if path.is_dir():
            path /= self.default_json_file_name
        with open(path, 'w') as f:
            data = dataclasses.asdict(self)
            # remove these two properties which are ephemeral and for the GUI
            del data['id']
            del data['task']
            # store the rest to file
            json.dump(data, f, cls=CustomTypeEncoder)

    @classmethod
    def load_from_json(cls, path: str | pathlib.Path):
        path = pathlib.Path(path)
        if path.is_dir():
            path /= cls.default_json_file_name
        with open(path, 'r') as f:
            return cls(**json.load(f, object_hook=json_reconstitute))


def make_fs_dirname(rec: Recording, dir: pathlib.Path = None):
    if rec.participant:
        dirname = f"{rec.eye_tracker.value}_{rec.participant}_{rec.name}"
    else:
        dirname = f"{rec.eye_tracker.value}_{rec.name}"

    # make sure its a valid path
    dirname = pathvalidate.sanitize_filename(dirname)

    # check it doesn't already exist
    if dir is not None:
        if (dir / dirname).is_dir():
            # add _1, _2, etc, until we find a unique name
            fver=1
            while (dir / f'{dirname}_{fver}').is_dir():
                fver += 1

            dirname = f'{dirname}_{fver}'

    return dirname

class CustomTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in [EyeTracker, Task, Status]:
            return {"__enum__": str(obj)}
        elif isinstance(obj,pathlib.Path):
            return {"__pathlib.Path__": str(obj)}
        elif isinstance(obj,Timestamp):
            return {"__Timestamp__": obj.value}
        return json.JSONEncoder.default(self, obj)

def json_reconstitute(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        match name:
            case 'EyeTracker':
                return getattr(EyeTracker, member)
            case 'Task':
                return getattr(Task, member)
            case 'Status':
                return getattr(Status, member)
            case other:
                raise ValueError(f'unknown enum "{other}"')
    elif "__pathlib.Path__" in d:
        return pathlib.Path(d["__pathlib.Path__"])
    elif "__Timestamp__" in d:
        return Timestamp(d["__Timestamp__"])
    else:
        return d


def getXYZLabels(stringList,N=3):
    if type(stringList) is not list:
        stringList = [stringList]
    return list(itertools.chain(*[[s+'_%s' % (chr(c)) for c in range(ord('x'), ord('x')+N)] for s in stringList]))

def noneIfAnyNan(vals):
    if not np.any(np.isnan(vals)):
        return vals
    else:
        return None

def dataReaderHelper(entry,lbl,N=3,type='float32'):
    columns = getXYZLabels(lbl,N)
    if np.all([x in entry for x in columns]):
        return np.array([entry[x] for x in columns]).astype(type)
    else:
        return None

def allNanIfNone(vals,numel):
    if vals is None:
        return np.array([math.nan for x in range(numel)])
    else:
        return vals

def cartesian_product(*arrays):
    ndim = len(arrays)
    return (np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim))

def av_rescale(a, b, c):
    # a * b / c, rounding to nearest and halfway cases away from zero
    # e.g., scale a from timebase c to timebase b
    # from ffmpeg libavutil mathematics.c, porting the simple case assuming that a, b and c <= INT_MAX
    r = c // 2
    return (a * b + r) // c

def getFrameTimestampsFromVideo(vid_file):
    """
    Parse the supplied video, return an array of frame timestamps. There must be only one video stream
    in the video file, because otherwise we do not know which is the correct stream.
    """
    if vid_file.suffix in ['.mov', '.mp4', '.m4a', '.3gp', '.3g2', '.mj2']:
        # parse mp4 file
        boxes       = iso.Mp4File(str(vid_file))
        summary     = boxes.get_summary()
        vid_tracks  = [t for t in summary['track_list'] if t['media_type']=='video']
        assert len(vid_tracks)==1, f"File has {len(vid_tracks)} video tracks (more than one), not supported"
        # 1. find mdat box
        moov        = boxes.children[[i for i,x in enumerate(boxes.children) if x.type=='moov'][0]]
        # 2. get global/movie time scale
        movie_time_scale = np.int64(moov.children[[i for i,x in enumerate(moov.children) if x.type=='mvhd'][0]].box_info['timescale'])
        # 3. find video track boxes
        trak_idxs   = [i for i,x in enumerate(moov.children) if x.type=='trak']
        trak_idxs   = [x for i,x in enumerate(trak_idxs) if summary['track_list'][i]['media_type']=='video']
        assert len(trak_idxs)==1
        trak        = moov.children[trak_idxs[0]]
        # 4. get mdia box
        mdia        = trak.children[[i for i,x in enumerate(trak.children) if x.type=='mdia'][0]]
        # 5. get media/track time_scale and duration fields from mdhd
        mdhd            = mdia.children[[i for i,x in enumerate(mdia.children) if x.type=='mdhd'][0]]
        media_time_scale= mdhd.box_info['timescale']
        # 6. check for presence of edit list
        # if its there, check its one we understand (one or multiple empty list at the beginning
        # to shift movie start, and/or a single non-empty edit list), and parse it. Else abort
        edts_idx= [i for i,x in enumerate(trak.children) if x.type=='edts']
        empty_duration  = np.int64(0)
        media_start_time= np.int64(-1)
        media_duration  = np.int64(-1)
        if edts_idx:
            elst = trak.children[edts_idx[0]].children[0]
            edit_start_index = 0
            # logic ported from mov_build_index()/mov_fix_index() in ffmpeg's libavformat/mov.c
            for i in range(elst.box_info['entry_count']):
                if i==edit_start_index and elst.box_info['entry_list'][i]['media_time'] == -1:
                    # empty edit list, indicates the start time of the stream
                    # relative to the presentation itself
                    this_empty_duration  = np.int64(elst.box_info['entry_list'][i]['segment_duration'])  # NB: in movie time scale
                    # convert duration from edit list from global timescale to track timescale
                    empty_duration  += av_rescale(this_empty_duration,media_time_scale,movie_time_scale)
                    edit_start_index+= 1
                elif i==edit_start_index and elst.box_info['entry_list'][i]['media_time'] > 0:
                    media_start_time = np.int64(elst.box_info['entry_list'][i]['media_time'])   # NB: already in track timescale, do not scale
                    media_duration   = av_rescale(np.int64(elst.box_info['entry_list'][i]['segment_duration']),media_time_scale,movie_time_scale)   # as above, scale to track timescale
                    if media_start_time<0:
                        raise RuntimeError('File contains an edit list that is too complicated (media start time < 0) for this parser, not supported')
                    if elst.box_info['entry_list'][i]['media_rate']!=1.0:
                        raise RuntimeError('File contains an edit list that is too complicated (media time is not 1.0) for this parser, not supported')
                elif i>edit_start_index:
                    raise RuntimeError('File contains an edit list that is too complicated (multiple non-empty edits) for this parser, not supported')
        # 7. get stbl
        minf    = mdia.children[[i for i,x in enumerate(mdia.children) if x.type=='minf'][0]]
        stbl    = minf.children[[i for i,x in enumerate(minf.children) if x.type=='stbl'][0]]
        # 8. check whether we have a ctts atom
        ctts_idx= [i for i,x in enumerate(stbl.children) if x.type=='ctts']
        if ctts_idx:
            ctts_ = stbl.children[ctts_idx[0]].box_info['entry_list']
            if any([e['sample_offset']<0 for e in ctts_]):
                # if we need to deal with them, we'd probably want to completely port mov_build_index() and mov_fix_index() in ffmpeg's libavformat/mov.c
                raise RuntimeError('Encountered a ctts (composition offset) atom with negative sample offsets, cannot handle that. aborting.')
            # uncompress
            total_frames_ctts = sum([e['sample_count'] for e in ctts_])
            ctts = np.zeros(total_frames_ctts, dtype=np.int64)
            idx = 0
            for e in ctts_:
                ctts[idx:idx+e['sample_count']] = e['sample_offset']
                idx = idx+e['sample_count']
        # 9. get sample table from stts
        stts = stbl.children[[i for i,x in enumerate(stbl.children) if x.type=='stts'][0]].box_info['entry_list']
        # uncompress the delta table
        total_frames_stts = sum([e['sample_count'] for e in stts])
        dts = np.zeros(total_frames_stts, dtype=np.int64)
        idx = 0
        for e in stts:
            dts[idx:idx+e['sample_count']] = e['sample_delta']
            idx = idx+e['sample_count']

        # 10. now put it all together
        # turn into timestamps
        dts = np.cumsum(np.insert(dts, 0, 0))
        dts = np.delete(dts,-1) # remove last, since that denotes _end_ of last frame, and we only need timestamps for frame onsets
        # apply ctts
        if ctts_idx:
            dts += ctts
            # ctts should lead to a reordering of frames, so sort
            dts = np.sort(dts)
        # if we have a non-empty edit list, apply
        if media_start_time!=-1:
            # remove all timestamps before start or after end of edit list
            to_keep = np.logical_or(dts >= media_start_time, dts < (media_duration + media_start_time))
            dts = dts[to_keep]
            min_corrected_pts = dts[0]  # already sorted, this is the first frame's pts
            # If there are empty edits, then min_corrected_pts might be positive
            # intentionally. So we subtract the sum duration of emtpy edits here.
            min_corrected_pts -= empty_duration
            # If the minimum pts turns out to be greater than zero,
            # then we subtract the dts by that amount to make the first pts zero.
            dts -= min_corrected_pts
        # now turn into timestamps in ms
        frameTs = (dts+empty_duration)/media_time_scale*1000
    else:
        # open file with opencv and get timestamps of each frame
        vid = cv2.VideoCapture(str(vid_file))
        nframes = float(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        frameTs = []
        frame_idx = 0
        while vid.isOpened():
            # read timestamp _before_ reading the frame, so we get position at start of the frame, not at
            # end
            ts = vid.get(cv2.CAP_PROP_POS_MSEC)

            ret, _ = vid.read()
            frame_idx += 1

            # You'd think to get the time at the start of the frame, which is what we want, you'd need to
            # read the time _before_ reading the frame. But there seems to be an off-by-one here for some
            # files, like at least some MP4s, but not in some AVIs in my testing. Catch this off-by-one
            # and to correct for it, do not store the first timestamp. This ensures we get a sensible
            # output (else first two frames have timestamp 0.0 ms).
            if frame_idx==1 and ts==vid.get(cv2.CAP_PROP_POS_MSEC):
                continue

            frameTs.append(ts)
            # check if we're done. Can't trust ret==False to indicate we're at end of video, as it may also return false for some frames when video has errors in the middle that we can just read past
            if (not ret and frame_idx>0 and frame_idx/nframes<.99):
                raise RuntimeError("The video file is corrupt. Testing has shown that it cannot be guaranteed that timestamps remain correct when trying to read past the hole. So abort, cannot process this video.")

        # release the video capture object
        vid.release()
        frameTs = np.array(frameTs)

    ### convert the frame_timestamps to dataframe
    frameIdx = np.arange(0, len(frameTs))
    frameTsDf = pd.DataFrame({'frame_idx': frameIdx, 'timestamp': frameTs})
    frameTsDf.set_index('frame_idx', inplace=True)

    return frameTsDf


class CV2VideoReader:
    def __init__(self, file: str|pathlib.Path, timestamps: list|np.ndarray|pd.DataFrame):
        self.file = pathlib.Path(file)
        if isinstance(timestamps,list):
            self.ts = np.array(timestamps)
        elif isinstance(timestamps, pd.DataFrame):
            self.ts = timestamps['timestamp'].to_numpy()
        else:
            self.ts = timestamps

        self.cap = cv2.VideoCapture(str(self.file))
        if not self.cap.isOpened():
            raise RuntimeError('the file "{}" could not be opened'.format(str(self.file)))
        self.nframes= float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = -1
        self._last_good_ts = (-1, -1., -1.)  # frame_idx, ts from opencv, ts from file
        self._is_off_by_one = False

    def __del__(self):
        self.cap.release()

    def get_prop(self, cv2_prop):
        return self.cap.get(cv2_prop)

    def set_prop(self, cv2_prop, val):
        return self.cap.set(cv2_prop, val)

    def read_frame(self):
        ts0 = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = self.cap.read()
        ts1 = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        self.frame_idx += 1

        # check if this is a stream for which opencv returns timestamps that are one frame off
        if self.frame_idx==0 and ts0==ts1:
            self._is_off_by_one = True
        # check if we're done. Can't trust ret==False to indicate we're at end of video, as
        # it may also return False for some corrupted frames that we can just read past
        if not ret and (self.frame_idx==0 or self.frame_idx/self.nframes>.99):
            return True, None, None, None

        # keep going
        ts_from_list = self.ts[self.frame_idx]
        if self.frame_idx==1 or ts1>0.:
            # check for gap, and if there is a gap, fix up frame_idx if needed
            if self._last_good_ts[0]!=-1 and ts_from_list-self._last_good_ts[2] < ts1-self._last_good_ts[1]-1:  # little bit of leeway for precision or mismatched timestamps
                # we skipped some frames altogether, need to correct current frame_idx
                t_jump = ts1-self._last_good_ts[1]
                tss = self.ts-self._last_good_ts[2]
                # find best matching frame idx so we catch up with the jump
                idx = bisect.bisect(tss, t_jump)
                if abs(tss[idx-1]-t_jump)<abs(tss[idx]-t_jump):
                    idx -= 1
                self.frame_idx = idx
                ts_from_list = self.ts[self.frame_idx]
            self._last_good_ts = (self.frame_idx, ts1, ts_from_list)

        # we might not have a valid frame, but we're not done yet
        if not ret or frame is None:
            return False, None,  self.frame_idx, ts_from_list
        else:
            return False, frame, self.frame_idx, ts_from_list

def tssToFrameNumber(ts,frameTimestamps,mode='nearest'):
    df = pd.DataFrame(index=ts)
    df.insert(0,'frame_idx',np.int64(0))

    # get index where this ts would be inserted into the frame_timestamp array
    idxs = np.searchsorted(frameTimestamps, ts)
    if mode=='after':
        idxs = idxs.astype('float32')
        # out of range, set to nan
        idxs[idxs==0] = np.nan
        # -1: since idx points to frame timestamp for frame after the one during which the ts ocurred, correct
        idxs -= 1
    elif mode=='nearest':
        # implementation from https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python/8929827#8929827
        # same logic as used by pupil labs
        idxs = np.clip(idxs, 1, len(frameTimestamps)-1)
        left = frameTimestamps[idxs-1]
        right = frameTimestamps[idxs]
        idxs -= ts - left < right - ts

    df=df.assign(frame_idx=idxs)
    if mode=='after':
        df=df.convert_dtypes() # turn into int64 again

    return df

class Marker:
    def __init__(self, key, center, corners=None, color=None, rot=0):
        self.key = key
        self.center = center
        self.corners = corners
        self.color = color
        self.rot = rot

    def __str__(self):
        ret = '[%d]: center @ (%.2f, %.2f), rot %.0f deg' % (self.key, self.center[0], self.center[1], self.rot)
        return ret

def corners_intersection(corners):
    line1 = ( corners[0], corners[2] )
    line2 = ( corners[1], corners[3] )
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array( [x,y] ).astype('float32')

def toNormPos(x,y,bbox):
    # transforms input (x,y) which is on a plane in world units
    # (e.g. mm on an aruco poster) to a normalized position
    # in an image of the plane, given the image's bounding box in
    # world units
    # for input (0,0) is bottom left, for output (0,0) is top left
    # bbox is [left, top, right, bottom]

    extents = [bbox[2]-bbox[0], bbox[1]-bbox[3]]
    pos     = [(x-bbox[0])/extents[0], (bbox[1]-y)/extents[1]]    # bbox[1]-y instead of y-bbox[3] to flip y
    return pos

def toImagePos(x,y,bbox,imSize,margin=[0,0]):
    # transforms input (x,y) which is on a plane in world units
    # (e.g. mm on an aruco poster) to a pixel position in the
    # image, given the image's bounding box in world units
    # imSize should be active image area in pixels, excluding margin

    # fractional position between bounding box edges, (0,0) in bottom left
    pos = toNormPos(x,y, bbox)
    # turn into int, add margin
    pos = [p*s+m for p,s,m in zip(pos,imSize,margin)]
    return pos

def arucoRefineDetectedMarkers(detector, image, arucoBoard, detectedCorners, detectedIds, rejectedCorners, cameraMatrix = None, distCoeffs= None):
    corners, ids, rejectedImgPoints, recoveredIds = detector.refineDetectedMarkers(
                            image = image, board = arucoBoard,
                            detectedCorners = detectedCorners, detectedIds = detectedIds, rejectedCorners = rejectedCorners,
                            cameraMatrix = cameraMatrix, distCoeffs = distCoeffs)
    if corners and corners[0].shape[0]==4:
        # there are versions out there where there is a bug in output shape of each set of corners, fix up
        corners = [np.reshape(c,(1,4,2)) for c in corners]
    if rejectedImgPoints and rejectedImgPoints[0].shape[0]==4:
        # same as for corners
        rejectedImgPoints = [np.reshape(c,(1,4,2)) for c in rejectedImgPoints]

    return corners, ids, rejectedImgPoints, recoveredIds


def estimateHomography(known, detectedCorners, detectedIDs):
    # collect matching corners in image and in world
    pts_src = []
    pts_dst = []
    detectedIDs = detectedIDs.flatten()
    if len(detectedIDs) != len(detectedCorners):
        raise ValueError('unexpected number of IDs (%d) given number of corner arrays (%d)' % (len(detectedIDs),len(detectedCorners)))
    for i in range(0, len(detectedIDs)):
        if detectedIDs[i] in known:
            dc = detectedCorners[i]
            if dc.shape[0]==1 and dc.shape[1]==4:
                dc = np.reshape(dc,(4,1,2))
            pts_src.extend( [x.flatten() for x in dc] )
            pts_dst.extend( known[detectedIDs[i]].corners )

    if len(pts_src) < 4:
        return None, False

    # compute Homography
    pts_src = np.float32(pts_src)
    pts_dst = np.float32(pts_dst)
    h, _ = cv2.findHomography(pts_src, pts_dst)

    return h, True

def applyHomography(h, x, y):
    if math.isnan(x) or math.isnan(y):
        return np.array([np.nan, np.nan])

    src = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    dst = cv2.perspectiveTransform(src,h)
    return dst.flatten()


def distortPoint(x, y, cameraMatrix, distCoeff):
    if math.isnan(x) or math.isnan(y):
        return np.array([np.nan, np.nan])

    # unproject, ignoring distortion as this is an undistored point
    points_2d = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    points_2d = cv2.undistortPoints(points_2d, cameraMatrix, np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]]))
    points_3d = cv2.convertPointsToHomogeneous(points_2d)
    points_3d.shape = -1, 3

    # reproject, applying distortion
    points_2d, _ = cv2.projectPoints(points_3d, np.zeros((1, 1, 3)), np.zeros((1, 1, 3)), cameraMatrix, distCoeff)

    return points_2d.flatten()

def undistortPoint(x, y, cameraMatrix, distCoeff):
    if math.isnan(x) or math.isnan(y):
        return np.array([np.nan, np.nan])

    points_2d = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    points_2d = cv2.undistortPoints(points_2d, cameraMatrix, distCoeff, P=cameraMatrix) # P=cameraMatrix to reproject to camera
    return points_2d.flatten()

def unprojectPoint(x, y, cameraMatrix, distCoeff):
    if math.isnan(x) or math.isnan(y):
        return np.array([np.nan, np.nan, np.nan])

    points_2d = np.asarray([x, y], dtype='float32').reshape((1, -1, 2))
    points_2d = cv2.undistortPoints(points_2d, cameraMatrix, distCoeff)
    points_3d = cv2.convertPointsToHomogeneous(points_2d)
    points_3d.shape = -1, 3
    return points_3d.flatten()


def angle_between(v1, v2): 
    return (180.0 / math.pi) * math.atan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2))

def intersect_plane_ray(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    # from https://rosettacode.org/wiki/Find_the_intersection_of_a_line_with_a_plane#Python

    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        # raise RuntimeError("no intersection or line is within plane")
        np.array([np.nan, np.nan, np.nan])

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    return w + si * rayDirection + planePoint


def drawOpenCVCircle(img, center_coordinates, radius, color, thickness, subPixelFac):
    p = [np.round(x*subPixelFac) for x in center_coordinates]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in p]):
        p = tuple([int(x) for x in p])
        cv2.circle(img, p, radius*subPixelFac, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))

def drawOpenCVLine(img, start_point, end_point, color, thickness, subPixelFac):
    sp = [np.round(x*subPixelFac) for x in start_point]
    ep = [np.round(x*subPixelFac) for x in   end_point]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in sp]) and np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in ep]):
        sp = tuple([int(x) for x in sp])
        ep = tuple([int(x) for x in ep])
        cv2.line(img, sp, ep, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))

def drawOpenCVRectangle(img, p1, p2, color, thickness, subPixelFac):
    p1 = [np.round(x*subPixelFac) for x in p1]
    p2 = [np.round(x*subPixelFac) for x in p2]
    if np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in p1]) and np.all([not math.isnan(x) and abs(x)<np.iinfo(int).max for x in p2]):
        p1 = tuple([int(x) for x in p1])
        p2 = tuple([int(x) for x in p2])
        cv2.rectangle(img, p1, p2, color, thickness, lineType=cv2.LINE_AA, shift=int(math.log2(subPixelFac)))

def drawOpenCVFrameAxis(img, cameraMatrix, distCoeffs, rvec,  tvec,  armLength, thickness, subPixelFac, position = [0.,0.,0.]):
    # same as the openCV function, but with anti-aliasing for a nicer image if subPixelFac>1
    points = np.vstack((np.zeros((1,3)), armLength*np.eye(3)))+np.vstack(4*[np.asarray(position)])
    points = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)[0]
    drawOpenCVLine(img, points[0].flatten(), points[1].flatten(), (0, 0, 255), thickness, subPixelFac)
    drawOpenCVLine(img, points[0].flatten(), points[2].flatten(), (0, 255, 0), thickness, subPixelFac)
    drawOpenCVLine(img, points[0].flatten(), points[3].flatten(), (255, 0, 0), thickness, subPixelFac)

def drawArucoDetectedMarkers(img,corners,ids,borderColor=(0,255,0), drawIDs = True, subPixelFac=1, specialHighlight = []):
    # same as the openCV function, but with anti-aliasing for a (much) nicer image if subPixelFac>1
    textColor   = [x for x in borderColor]
    cornerColor = [x for x in borderColor]
    textColor[0]  , textColor[1]   = textColor[1]  , textColor[0]       #   text color just swap B and R
    cornerColor[1], cornerColor[2] = cornerColor[2], cornerColor[1]     # corner color just swap G and R

    drawIDs = drawIDs and (ids is not None) and len(ids)>0

    for i in range(0, len(corners)):
        corner = corners[i][0]
        # draw marker sides
        sideColor = borderColor
        for s,c in zip(specialHighlight[::2],specialHighlight[1::2]):
            if s is not None and ids[i][0] in s:
                sideColor = c
        for j in range(4):
            p0 = corner[j,:]
            p1 = corner[(j + 1) % 4,:]
            drawOpenCVLine(img, p0, p1, sideColor, 1, subPixelFac)

        # draw first corner mark
        p1 = corner[0]
        drawOpenCVRectangle(img, corner[0]-3, corner[0]+3, cornerColor, 1, subPixelFac)

        # draw IDs if wanted
        if drawIDs:
            c = corners_intersection(corner)
            cv2.putText(img, str(ids[i][0]), tuple(c.astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2, lineType=cv2.LINE_AA)



class Gaze:
    def __init__(self, ts, vid2D, world3D=None, lGazeVec=None, lGazeOrigin=None, rGazeVec=None, rGazeOrigin=None):
        self.ts = ts
        self.vid2D = vid2D              # gaze point on the scene video
        self.world3D = world3D          # gaze point in the world (often binocular gaze point)
        self.lGazeVec= lGazeVec
        self.lGazeOrigin = lGazeOrigin
        self.rGazeVec= rGazeVec
        self.rGazeOrigin = rGazeOrigin

    @staticmethod
    def readDataFromFile(fileName):
        gazes = {}
        maxFrameIdx = 0
        with open(str(fileName), 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                frame_idx   = float(entry['frame_idx'])
                ts          = float(entry['timestamp'])

                vid2D       = dataReaderHelper(entry,'vid_gaze_pos',2)
                world3D     = dataReaderHelper(entry,'3d_gaze_pos')
                lGazeVec    = dataReaderHelper(entry,'l_gaze_dir')
                lGazeOrigin = dataReaderHelper(entry,'l_gaze_ori')
                rGazeVec    = dataReaderHelper(entry,'r_gaze_dir')
                rGazeOrigin = dataReaderHelper(entry,'r_gaze_ori')
                gaze = Gaze(ts, vid2D, world3D, lGazeVec, lGazeOrigin, rGazeVec, rGazeOrigin)

                if frame_idx in gazes:
                    gazes[frame_idx].append(gaze)
                else:
                    gazes[frame_idx] = [gaze]

                maxFrameIdx = int(max(maxFrameIdx,frame_idx))

        return gazes,maxFrameIdx

    def draw(self, img, subPixelFac=1, camRot=None, camPos=None, cameraMatrix=None, distCoeff=None):
        drawOpenCVCircle(img, self.vid2D, 8, (0,255,0), 2, subPixelFac)
        # draw 3D gaze point as well, usually coincides with 2D gaze point, but not always. E.g. the Adhawk MindLink may
        # apply a correction for parallax error to the projected gaze point using the vergence signal.
        if self.world3D is not None and camRot is not None and camPos is not None and cameraMatrix is not None and distCoeff is not None:
            a = cv2.projectPoints(np.array(self.world3D).reshape(1,3),camRot,camPos,cameraMatrix,distCoeff)[0][0][0]
            drawOpenCVCircle(img, a, 5, (0,255,255), -1, subPixelFac)


def getMarkerUnrotated(cornerPoints, rot):
    # markers are rotated in multiples of 90 only, so can easily unrotate
    if rot == -90:
        # -90 deg
        cornerPoints = np.vstack((cornerPoints[-1,:], cornerPoints[0:3,:]))
    elif rot == 90:
        # 90 deg
        cornerPoints = np.vstack((cornerPoints[1:,:], cornerPoints[0,:]))
    elif rot == 180:
        # 180 deg
        cornerPoints = np.vstack((cornerPoints[2:,:], cornerPoints[0:2,:]))

    return cornerPoints

class Poster:
    posterImageFilename = 'referencePoster.png'
    aruco_dict          = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

    def __init__(self, configDir, validationSetup, imHeight = 400):
        if configDir is not None:
            configDir = pathlib.Path(configDir)

        # get marker width
        if validationSetup['mode'] == 'deg':
            self.cellSizeMm = 2.*math.tan(math.radians(.5))*validationSetup['distance']*10
        else:
            self.cellSizeMm = 10 # 1cm
        self.markerSize = self.cellSizeMm*validationSetup['markerSide']

        # get information about poster
        self._getTargetsAndKnownMarkers(configDir, validationSetup)

        # get image of poster
        useTempDir = configDir is None
        if useTempDir:
            tempDir = tempfile.TemporaryDirectory()
            configDir = pathlib.Path(tempDir.name)

        posterImage = configDir / self.posterImageFilename
        # 1 if doesn't exist, create
        if not posterImage.is_file():
            self._storeReferencePoster(posterImage, validationSetup)
        # 2. read image
        self.img = cv2.imread(str(posterImage), cv2.IMREAD_COLOR)

        if useTempDir:
            tempDir.cleanup()

        if imHeight==-1:
            self.scale = 1
        else:
            self.scale = float(imHeight)/self.img.shape[0]
            self.img = cv2.resize(self.img, None, fx=self.scale, fy=self.scale, interpolation = cv2.INTER_AREA)
        self.height, self.width, self.channels = self.img.shape

    def getImgCopy(self, asRGB=False):
        if asRGB:
            return self.img[:,:,[2,1,0]]    # indexing returns a copy
        else:
            return self.img.copy()

    def draw(self, img, x, y, subPixelFac=1, color=None, size=6):
        if not math.isnan(x):
            xy = toImagePos(x,y,self.bbox,[self.width, self.height])
            if color is None:
                drawOpenCVCircle(img, xy, 8, (0,255,0), -1, subPixelFac)
                color = (0,0,0)
            drawOpenCVCircle(img, xy, size, color, -1, subPixelFac)

    def _getTargetsAndKnownMarkers(self, config_dir, validationSetup):
        """ poster space: (0,0) is at center target, (-,-) bottom left """

        # read in target positions
        self.targets = {}
        targets = gv_config.get_targets(config_dir, validationSetup['targetPosFile'])
        if targets is not None:
            center  = targets.loc[validationSetup['centerTarget'],['x','y']]
            targets.x = self.cellSizeMm * (targets.x.astype('float32') - center.x)
            targets.y = self.cellSizeMm * (targets.y.astype('float32') - center.y)
            for idx, row in targets.iterrows():
                self.targets[idx] = Marker(idx, row[['x','y']].values, color=row.color)
        else:
            center = pd.Series(data=[0.,0.],index=['x','y'])


        # read in aruco marker positions
        markerHalfSizeMm  = self.markerSize/2.
        self.knownMarkers = {}
        self.bbox         = []
        markerPos = gv_config.get_markers(config_dir, validationSetup['markerPosFile'])
        if markerPos is not None:
            markerPos.x = self.cellSizeMm * (markerPos.x.astype('float32') - center.x)
            markerPos.y = self.cellSizeMm * (markerPos.y.astype('float32') - center.y)
            for idx, row in markerPos.iterrows():
                c   = row[['x','y']].values
                # rotate markers (negative because poster coordinate system)
                rot = row[['rotation_angle']].values[0]
                if rot%90 != 0:
                    raise ValueError("Rotation of a marker must be a multiple of 90 degrees")
                rotr= -math.radians(rot)
                R   = np.array([[math.cos(rotr), math.sin(rotr)], [-math.sin(rotr), math.cos(rotr)]])
                # top left first, and clockwise: same order as detected aruco marker corners
                tl = c + np.matmul(R,np.array( [ -markerHalfSizeMm , -markerHalfSizeMm ] ))
                tr = c + np.matmul(R,np.array( [  markerHalfSizeMm , -markerHalfSizeMm ] ))
                br = c + np.matmul(R,np.array( [  markerHalfSizeMm ,  markerHalfSizeMm ] ))
                bl = c + np.matmul(R,np.array( [ -markerHalfSizeMm ,  markerHalfSizeMm ] ))

                self.knownMarkers[idx] = Marker(idx, c, corners=[ tl, tr, br, bl ], rot=rot)

            # determine bounding box of markers ([left, top, right, bottom])
            # NB: this assumes that poster has an outer edge of markers, i.e.,
            # that it does not have targets at its edges. Also assumes markers
            # are rotated by multiples of 90 degrees
            self.bbox.append(markerPos.x.min()-markerHalfSizeMm)
            self.bbox.append(markerPos.y.min()-markerHalfSizeMm)
            self.bbox.append(markerPos.x.max()+markerHalfSizeMm)
            self.bbox.append(markerPos.y.max()+markerHalfSizeMm)

    def getArucoBoard(self, unRotateMarkers=False):
        boardCornerPoints = []
        ids = []
        for key in self.knownMarkers:
            ids.append(key)
            cornerPoints = np.vstack(self.knownMarkers[key].corners).astype('float32')
            if unRotateMarkers:
                cornerPoints = getMarkerUnrotated(cornerPoints,self.knownMarkers[key].rot)

            boardCornerPoints.append(cornerPoints)

        boardCornerPoints = np.dstack(boardCornerPoints)        # list of 2D arrays -> 3D array
        boardCornerPoints = np.rollaxis(boardCornerPoints,-1)   # 4x2xN -> Nx4x2
        boardCornerPoints = np.pad(boardCornerPoints,((0,0),(0,0),(0,1)),'constant', constant_values=(0.,0.)) # Nx4x2 -> Nx4x3
        return cv2.aruco.Board(boardCornerPoints, self.aruco_dict, np.array(ids))

    def _storeReferencePoster(self, posterImage, validationSetup):
        referenceBoard = self.getArucoBoard(unRotateMarkers = True)
        # get image with markers
        bboxExtents    = [self.bbox[2]-self.bbox[0], math.fabs(self.bbox[3]-self.bbox[1])]  # math.fabs to deal with bboxes where (-,-) is bottom left
        aspectRatio    = bboxExtents[0]/bboxExtents[1]
        refBoardWidth  = validationSetup['referencePosterWidth']
        refBoardHeight = math.ceil(refBoardWidth/aspectRatio)
        margin         = 1  # always 1 pixel, anything else behaves strangely (markers are drawn over margin as well)

        refBoardImage  = cv2.cvtColor(
            referenceBoard.generateImage(
                (refBoardWidth+2*margin,refBoardHeight+2*margin),margin,validationSetup['markerBorderBits']),
            cv2.COLOR_GRAY2RGB
        )
        # cut off this 1-pix margin
        assert refBoardImage.shape[0]==refBoardHeight+2*margin,"Output image height is not as expected"
        assert refBoardImage.shape[1]==refBoardWidth +2*margin,"Output image width is not as expected"
        refBoardImage  = refBoardImage[1:-1,1:-1,:]
        # walk through all markers, if any are supposed to be rotated, do so
        minX =  np.inf
        maxX = -np.inf
        minY =  np.inf
        maxY = -np.inf
        rots = []
        cornerPointsU = []
        for key in self.knownMarkers:
            cornerPoints = np.vstack(self.knownMarkers[key].corners).astype('float32')
            cornerPointsU.append(getMarkerUnrotated(cornerPoints, self.knownMarkers[key].rot))
            rots.append(self.knownMarkers[key].rot)
            minX = np.min(np.hstack((minX,cornerPoints[:,0])))
            maxX = np.max(np.hstack((maxX,cornerPoints[:,0])))
            minY = np.min(np.hstack((minY,cornerPoints[:,1])))
            maxY = np.max(np.hstack((maxY,cornerPoints[:,1])))
        if np.any(np.array(rots)!=0):
            # determine where the markers are placed
            sizeX = maxX - minX
            sizeY = maxY - minY
            xReduction = sizeX / float(refBoardImage.shape[1])
            yReduction = sizeY / float(refBoardImage.shape[0])
            if xReduction > yReduction:
                nRows = int(sizeY / xReduction);
                yMargin = (refBoardImage.shape[0] - nRows) / 2;
                xMargin = 0
            else:
                nCols = int(sizeX / yReduction);
                xMargin = (refBoardImage.shape[1] - nCols) / 2;
                yMargin = 0

            for r,cpu in zip(rots,cornerPointsU):
                if r != 0:
                    # figure out where marker is
                    cpu -= np.array([[minX,minY]])
                    cpu[:,0] =       cpu[:,0] / sizeX  * float(refBoardImage.shape[1]) + xMargin
                    cpu[:,1] = (1. - cpu[:,1] / sizeY) * float(refBoardImage.shape[0]) + yMargin
                    sz = np.min(cpu[2,:]-cpu[0,:])
                    # get marker
                    cpu = np.floor(cpu)
                    idxs = np.floor([cpu[0,1], cpu[0,1]+sz, cpu[0,0], cpu[0,0]+sz]).astype('int')
                    marker = refBoardImage[idxs[0]:idxs[1], idxs[2]:idxs[3]]
                    # rotate (opposite because coordinate system) and put back
                    if r==-90:
                        marker = cv2.rotate(marker, cv2.ROTATE_90_CLOCKWISE)
                    elif r==90:
                        marker = cv2.rotate(marker, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif r==180:
                        marker = cv2.rotate(marker, cv2.ROTATE_180)

                    refBoardImage[idxs[0]:idxs[1], idxs[2]:idxs[3]] = marker

        # add targets
        subPixelFac = 8   # for sub-pixel positioning
        for key in self.targets:
            # 1. determine position on image
            circlePos = toImagePos(*self.targets[key].center, self.bbox,[refBoardWidth,refBoardHeight])

            # 2. draw
            clr = tuple([int(i*255) for i in colors.to_rgb(self.targets[key].color)[::-1]])  # need BGR color ordering
            drawOpenCVCircle(refBoardImage, circlePos, 15, clr, -1, subPixelFac)

        cv2.imwrite(str(posterImage), refBoardImage)

class GazePoster:
    def __init__(self, ts, gaze3DRay=None, gaze3DHomography=None, wGaze3D=None, lGazeOrigin=None, lGaze3D=None, rGazeOrigin=None, rGaze3D=None, gaze2DRay=None, gaze2DHomography=None, wGaze2D=None, lGaze2D=None, rGaze2D=None):
        # 3D gaze is in world space, w.r.t. scene camera
        # 2D gaze is on the poster
        self.ts = ts

        # in camera space (3D coordinates)
        self.gaze3DRay        = gaze3DRay           # video gaze position on plane (camera ray intersected with plane)
        self.gaze3DHomography = gaze3DHomography    # gaze2DHomography in camera space
        self.wGaze3D          = wGaze3D             # 3D gaze point on plane (world-space gaze point, turned into direction ray and intersected with plane)
        self.lGazeOrigin      = lGazeOrigin
        self.lGaze3D          = lGaze3D             # 3D gaze point on plane ( left eye gaze vector intersected with plane)
        self.rGazeOrigin      = rGazeOrigin
        self.rGaze3D          = rGaze3D             # 3D gaze point on plane (right eye gaze vector intersected with plane)

        # in poster space (2D coordinates)
        self.gaze2DRay        = gaze2DRay           # Video gaze point mapped to poster by turning into direction ray and intersecting with poster
        self.gaze2DHomography = gaze2DHomography    # Video gaze point directly mapped to poster through homography transformation
        self.wGaze2D          = wGaze2D             # wGaze3D in poster space
        self.lGaze2D          = lGaze2D             # lGaze3D in poster space
        self.rGaze2D          = rGaze2D             # rGaze3D in poster space

    @staticmethod
    def getWriteHeader():
        header = ['gaze_timestamp']
        header.extend(getXYZLabels(['gazePosCam_vidPos_ray','gazePosCam_vidPos_homography']))
        header.extend(getXYZLabels(['gazePosCamWorld']))
        header.extend(getXYZLabels(['gazeOriCamLeft','gazePosCamLeft']))
        header.extend(getXYZLabels(['gazeOriCamRight','gazePosCamRight']))
        header.extend(getXYZLabels('gazePosPoster2D_vidPos_ray',2))
        header.extend(getXYZLabels('gazePosPoster2D_vidPos_homography',2))
        header.extend(getXYZLabels('gazePosPoster2DWorld',2))
        header.extend(getXYZLabels('gazePosPoster2DLeft',2))
        header.extend(getXYZLabels('gazePosPoster2DRight',2))
        return header

    @staticmethod
    def getMissingWriteData():
        return [math.nan for x in range(29)]

    def getWriteData(self):
        writeData = [self.ts]
        # in camera space
        writeData.extend(allNanIfNone(self.gaze3DRay,3))
        writeData.extend(allNanIfNone(self.gaze3DHomography,3))
        writeData.extend(allNanIfNone(self.wGaze3D,3))
        writeData.extend(allNanIfNone(self.lGazeOrigin,3))
        writeData.extend(allNanIfNone(self.lGaze3D,3))
        writeData.extend(allNanIfNone(self.rGazeOrigin,3))
        writeData.extend(allNanIfNone(self.rGaze3D,3))
        # in poster space
        writeData.extend(allNanIfNone(self.gaze2DRay,2))
        writeData.extend(allNanIfNone(self.gaze2DHomography,2))
        writeData.extend(allNanIfNone(self.wGaze2D,2))
        writeData.extend(allNanIfNone(self.lGaze2D,2))
        writeData.extend(allNanIfNone(self.rGaze2D,2))

        return writeData

    @staticmethod
    def readDataFromFile(fileName,start=None,end=None,stopOnceExceeded=False):
        gazes = {}
        readSubset = start is not None and end is not None
        with open(str(fileName), 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                frame_idx = int(entry['frame_idx'])
                if readSubset and (frame_idx<start or frame_idx>end):
                    if stopOnceExceeded and frame_idx>end:
                        break
                    else:
                        continue

                ts = float(entry['gaze_timestamp'])
                gaze3DRay       = dataReaderHelper(entry,'gazePosCam_vidPos_ray')
                gaze3DHomography= dataReaderHelper(entry,'gazePosCam_vidPos_homography')
                wGaze3D         = dataReaderHelper(entry,'gazePosCamWorld')
                lGazeOrigin     = dataReaderHelper(entry,'gazeOriCamLeft')
                lGaze3D         = dataReaderHelper(entry,'gazePosCamLeft')
                rGazeOrigin     = dataReaderHelper(entry,'gazeOriCamRight')
                rGaze3D         = dataReaderHelper(entry,'gazePosCamRight')
                gaze2DRay       = dataReaderHelper(entry,'gazePosPoster2D_vidPos_ray',2)
                gaze2DHomography= dataReaderHelper(entry,'gazePosPoster2D_vidPos_homography',2)
                wGaze2D         = dataReaderHelper(entry,'gazePosPoster2DWorld',2)
                lGaze2D         = dataReaderHelper(entry,'gazePosPoster2DLeft',2)
                rGaze2D         = dataReaderHelper(entry,'gazePosPoster2DRight',2)
                gaze = GazePoster(ts, gaze3DRay, gaze3DHomography, wGaze3D, lGazeOrigin, lGaze3D, rGazeOrigin, rGaze3D, gaze2DRay, gaze2DHomography, wGaze2D, lGaze2D, rGaze2D)

                if frame_idx in gazes:
                    gazes[frame_idx].append(gaze)
                else:
                    gazes[frame_idx] = [gaze]

        return gazes

    def drawOnWorldVideo(self, img, cameraMatrix, distCoeff, subPixelFac=1):
        # project to camera, display
        # gaze ray
        if self.gaze3DRay is not None:
            pPointCam = cv2.projectPoints(self.gaze3DRay.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            drawOpenCVCircle(img, pPointCam, 3, (255,255,0), -1, subPixelFac)
        # binocular gaze point
        if self.wGaze3D is not None:
            pPointCam = cv2.projectPoints(self.wGaze3D.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            drawOpenCVCircle(img, pPointCam, 3, (255,0,255), -1, subPixelFac)
        # left eye
        if self.lGaze3D is not None:
            pPointCam = cv2.projectPoints(self.lGaze3D.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            drawOpenCVCircle(img, pPointCam, 3, (0,0,255), -1, subPixelFac)
        # right eye
        if self.rGaze3D is not None:
            pPointCam = cv2.projectPoints(self.rGaze3D.reshape(1,3),np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            drawOpenCVCircle(img, pPointCam, 3, (255,0,0), -1, subPixelFac)
        # average
        if (self.lGaze3D is not None) and (self.rGaze3D is not None):
            pointCam  = np.array([(x+y)/2 for x,y in zip(self.lGaze3D,self.rGaze3D)]).reshape(1,3)
            pPointCam = cv2.projectPoints(pointCam,np.zeros((1,3)),np.zeros((1,3)),cameraMatrix,distCoeff)[0][0][0]
            if not math.isnan(pPointCam[0]):
                drawOpenCVCircle(img, pPointCam, 6, (255,0,255), -1, subPixelFac)

    def drawOnPoster(self, img, reference, subPixelFac=1):
        # binocular gaze point
        if self.wGaze2D is not None:
            reference.draw(img, self.wGaze2D[0],self.wGaze2D[1], subPixelFac, (0,255,255), 3)
        # left eye
        if self.lGaze2D is not None:
            reference.draw(img, self.lGaze2D[0],self.lGaze2D[1], subPixelFac, (0,0,255), 3)
        # right eye
        if self.rGaze2D is not None:
            reference.draw(img, self.rGaze2D[0],self.rGaze2D[1], subPixelFac, (255,0,0), 3)
        # average
        if (self.lGaze2D is not None) and (self.rGaze2D is not None):
            average = np.array([(x+y)/2 for x,y in zip(self.lGaze2D,self.rGaze2D)])
            if not math.isnan(average[0]):
                reference.draw(img, average[0], average[1], subPixelFac, (255,0,255))
        # video gaze position
        if self.gaze2DHomography is not None:
            reference.draw(img, self.gaze2DHomography[0],self.gaze2DHomography[1], subPixelFac, (0,255,0), 5)
        if self.gaze2DRay is not None:
            reference.draw(img, self.gaze2DRay[0],self.gaze2DRay[1], subPixelFac, (255,255,0), 3)

class PosterPose:
    def __init__(self, frameIdx, poseOk=False, nMarkers=0, rVec=None, tVec=None, hMat=None):
        self.frameIdx   = frameIdx

        # pose
        self.poseOk     = poseOk    # Output of cv2.SolvePnP(), whether successful or not
        self.nMarkers   = nMarkers  # number of Aruco markers this pose estimate is based on
        self.rVec       = rVec
        self.tVec       = tVec

        # homography
        self.hMat       = hMat.reshape(3,3) if hMat is not None else hMat

        # internals
        self._RMat        = None
        self._RtMat       = None
        self._planeNormal = None
        self._planePoint  = None
        self._RMatInv     = None
        self._RtMatInv    = None

    @staticmethod
    def getWriteHeader():
        header = ['frame_idx','poseOk','poseNMarker']
        header.extend(getXYZLabels(['poseRvec','poseTvec']))
        header.extend(['homography[%d,%d]' % (r,c) for r in range(3) for c in range(3)])
        return header

    @staticmethod
    def getMissingWriteData():
        dat = [False,0]
        dat.extend([math.nan for x in range(15)])
        return dat

    def getWriteData(self):
        writeData = [self.frameIdx]
        if not self.poseOk:
            writeData.extend(self.getMissingWriteData())
        else:
            # in camera space
            writeData.extend([True, self.nMarkers])
            writeData.extend(allNanIfNone(self.rVec,3).flatten())
            writeData.extend(allNanIfNone(self.tVec,3).flatten())
            writeData.extend(allNanIfNone(self.hMat,9).flatten())

        return writeData

    @staticmethod
    def readDataFromFile(fileName,start=None,end=None,stopOnceExceeded=False):
        poses       = {}
        readSubset  = start is not None and end is not None
        data        = pd.read_csv(str(fileName), delimiter='\t',index_col=False)
        rCols       = [col for col in data.columns if 'poseRvec' in col]
        tCols       = [col for col in data.columns if 'poseTvec' in col]
        hCols       = [col for col in data.columns if 'homography' in col]
        # run through all columns
        for idx, row in data.iterrows():
            frame_idx = int(row['frame_idx'])
            if readSubset and (frame_idx<start or frame_idx>end):
                if stopOnceExceeded and frame_idx>end:
                    break
                else:
                    continue

            # get all values (None if all nan)
            args = tuple(noneIfAnyNan(row[c].to_numpy().astype('float')) for c in (rCols,tCols,hCols))

            # insert if any non-None
            if not np.all([x is None for x in args]):   # check for not all isNone
                poses[frame_idx] = PosterPose(frame_idx,bool(row['poseOk']),int(row['poseNMarker']),*args)

        return poses

    def camToWorld(self, point):
        if (self.rVec is None) or (self.tVec is None) or np.any(np.isnan(point)):
            return np.array([np.nan, np.nan, np.nan])

        if self._RtMatInv is None:
            if self._RMatInv is None:
                if self._RMat is None:
                    self._RMat = cv2.Rodrigues(self.rVec)[0]
                self._RMatInv = self._RMat.T
            self._RtMatInv = np.hstack((self._RMatInv,np.matmul(-self._RMatInv,self.tVec.reshape(3,1))))

        return np.matmul(self._RtMatInv,np.append(np.array(point),1.).reshape((4,1))).flatten()

    def worldToCam(self, point):
        if (self.rVec is None) or (self.tVec is None) or np.any(np.isnan(point)):
            return np.array([np.nan, np.nan, np.nan])

        if self._RtMat is None:
            if self._RMat is None:
                self._RMat = cv2.Rodrigues(self.rVec)[0]
            self._RtMat = np.hstack((self._RMat, self.tVec.reshape(3,1)))

        return np.matmul(self._RtMat,np.append(np.array(point),1.).reshape((4,1))).flatten()

    def vectorIntersect(self, vector, origin = np.array([0.,0.,0.])):
        if (self.rVec is None) or (self.tVec is None) or np.any(np.isnan(vector)):
            return np.array([np.nan, np.nan, np.nan])

        if self._planeNormal is None:
            if self._RtMat is None:
                if self._RMat is None:
                    self._RMat = cv2.Rodrigues(self.rVec)[0]
                self._RtMat = np.hstack((self._RMat, self.tVec.reshape(3,1)))

            # get poster normal
            self._planeNormal = np.matmul(self._RMat, np.array([0., 0., 1.]))
            # get point on poster (just use origin)
            self._planePoint  = np.matmul(self._RtMat, np.array([0., 0., 0., 1.]))


        # normalize vector
        vector /= np.sqrt((vector**2).sum())

        # find intersection of 3D gaze with poster
        return intersect_plane_ray(self._planeNormal, self._planePoint, vector.flatten(), origin.flatten())

def get_timestamps_from_file(file) -> np.ndarray:
    return np.genfromtxt(file, dtype=None, delimiter='\t', skip_header=1, usecols=1)

class Idx2Timestamp:
    def __init__(self, fileName):
        self.timestamps = {}
        with open(str(fileName), 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                frame_idx = int(float(entry['frame_idx']))
                if frame_idx!=-1:
                    self.timestamps[frame_idx] = float(entry['timestamp'])

    def get(self, idx):
        if idx in self.timestamps:
            return self.timestamps[int(idx)]
        else:
            warnings.warn('frame_idx %d is not in set\n' % ( idx ), RuntimeWarning )
            return -1.

class Timestamp2Index:
    def __init__(self, fileName):
        self.indices = []
        self.timestamps = []
        with open(str(fileName), 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                frame_idx = int(float(entry['frame_idx']))
                if frame_idx!=-1:
                    self.indices   .append(int(float(entry['frame_idx'])))
                    self.timestamps.append(    float(entry['timestamp']))

    def find(self, ts):
        idx = bisect.bisect(self.timestamps, ts)
        # return nearest
        if idx>=len(self.timestamps):
            return self.indices[-1]
        elif idx>0 and abs(self.timestamps[idx-1]-ts)<abs(self.timestamps[idx]-ts):
            return self.indices[idx-1]
        else:
            return self.indices[idx]

    def getLast(self):
        return self.indices[-1], self.timestamps[-1]

    def getIFI(self):
        return np.mean(np.diff(self.timestamps))

def readCameraCalibrationFile(fileName):
    fs = cv2.FileStorage(str(fileName), cv2.FILE_STORAGE_READ)
    cameraMatrix    = fs.getNode("cameraMatrix").mat()
    distCoeff       = fs.getNode("distCoeff").mat()
    # camera extrinsics for 3D gaze
    cameraRotation  = fs.getNode("rotation").mat()
    if cameraRotation is not None:
        cameraRotation  = cv2.Rodrigues(cameraRotation)[0]  # need rotation vector, not rotation matrix
    cameraPosition  = fs.getNode("position").mat()
    fs.release()

    return (cameraMatrix,distCoeff,cameraRotation,cameraPosition)

def readMarkerIntervalsFile(fileName):
    analyzeFrames = []
    if pathlib.Path(fileName).is_file():
        with open(str(fileName), 'r' ) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for entry in reader:
                analyzeFrames.append(int(float(entry['start_frame'])))
                analyzeFrames.append(int(float(entry['end_frame'])))

    return None if len(analyzeFrames)==0 else analyzeFrames

def gazeToPlane(gaze,posterPose,cameraRotation,cameraPosition, cameraMatrix=None, distCoeffs=None):
    hasCameraPose = (posterPose.rVec is not None) and (posterPose.tVec is not None)
    gazePoster    = GazePoster(gaze.ts)
    if hasCameraPose:
        # get transform from ET data's coordinate frame to camera's coordinate frame
        if cameraRotation is None:
            cameraRotation = np.zeros((3,1))
        RCam  = cv2.Rodrigues(cameraRotation)[0]
        if cameraPosition is None:
            cameraPosition = np.zeros((3,1))
        RtCam = np.hstack((RCam, cameraPosition))

        # project gaze on video to reference poster using camera pose
        # turn observed gaze position on video into position on tangent plane
        g3D = unprojectPoint(gaze.vid2D[0],gaze.vid2D[1],cameraMatrix,distCoeffs)

        # find intersection of 3D gaze with poster
        gazePoster.gaze3DRay = posterPose.vectorIntersect(g3D)  # default vec origin (0,0,0) because we use g3D from camera's view point

        # above intersection is in camera space, turn into poster space to get position on poster
        (x,y,z)   = posterPose.camToWorld(gazePoster.gaze3DRay) # z should be very close to zero
        gazePoster.gaze2DRay = [x, y]

        # project world-space gaze point (often binocular gaze point) to plane
        if gaze.world3D is not None:
            # transform 3D gaze point from eye tracker space to camera space
            g3D = np.matmul(RtCam,np.array(np.append(gaze.world3D, 1)).reshape(4,1))

            # find intersection with poster (NB: pose is in camera reference frame)
            gazePoster.wGaze3D = posterPose.vectorIntersect(g3D)    # default vec origin (0,0,0) is fine because we work from camera's view point

            # above intersection is in camera space, turn into poster space to get position on poster
            (x,y,z)   = posterPose.camToWorld(gazePoster.wGaze3D)   # z should be very close to zero
            gazePoster.wGaze2D = [x, y]

    # unproject 2D gaze point on video to point on poster (should yield values very close to
    # the above method of intersecting video gaze point ray with poster, and usually also very
    # close to binocular gaze point (though for at least one tracker the latter is not the case;
    # the AdHawk has an optional parallax correction using a vergence signal))
    if posterPose.hMat is not None:
        ux, uy   = gaze.vid2D
        if (cameraMatrix is not None) and (distCoeffs is not None):
            ux, uy   = undistortPoint( ux, uy, cameraMatrix, distCoeffs)
        (xW, yW) = applyHomography(posterPose.hMat, ux, uy)
        gazePoster.gaze2DHomography = [xW, yW]

        # get this point in camera space
        if hasCameraPose:
            gazePoster.gaze3DHomography = posterPose.worldToCam(np.array([xW,yW,0.]))

    # project gaze vectors to reference poster (and draw on video)
    if not hasCameraPose:
        # nothing to do anymore
        return gazePoster

    gazeVecs    = [gaze.lGazeVec   , gaze.rGazeVec]
    gazeOrigins = [gaze.lGazeOrigin, gaze.rGazeOrigin]
    attrs       = [['lGazeOrigin','lGaze3D','lGaze2D'],['rGazeOrigin','rGaze3D','rGaze2D']]
    for gVec,gOri,attr in zip(gazeVecs,gazeOrigins,attrs):
        if gVec is None or gOri is None:
            continue
        # get gaze vector and point on vector (origin, e.g. pupil center) ->
        # transform from ET data coordinate frame into camera coordinate frame
        gVec    = np.matmul(RCam ,          gVec    )
        gOri    = np.matmul(RtCam,np.append(gOri,1.))
        setattr(gazePoster,attr[0],gOri)

        # intersect with poster -> yield point on poster in camera reference frame
        gPoster = posterPose.vectorIntersect(gVec, gOri)
        setattr(gazePoster,attr[1],gPoster)

        # transform intersection with poster from camera space to poster space
        (x,y,z)  = posterPose.camToWorld(gPoster)  # z should be very close to zero
        setattr(gazePoster,attr[2],[x, y])

    return gazePoster

def selectDictRange(theDict,start,end):
    return {k: theDict[k] for k in theDict if k>=start and k<=end}


__all__ = ['make_video','EyeTracker','Recording']