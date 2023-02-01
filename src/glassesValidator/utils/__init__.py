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
    Pupil_Core      = auto()
    Pupil_Invisible = auto()
    SMI_ETG         = auto()
    SeeTrue         = auto()
    Tobii_Glasses_2 = auto()
    Tobii_Glasses_3 = auto()
    Unknown         = auto()
eye_tracker_names = [x.value for x in EyeTracker if x!=EyeTracker.Unknown]

EyeTracker.Pupil_Core     .color = hex_to_rgba_0_1("#E6194B")
EyeTracker.Pupil_Invisible.color = hex_to_rgba_0_1("#3CB44B")
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
            return 'Code'
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


def update_recording_status(path: str | pathlib.Path, task: Task, status: Status):
    rec_status = get_recording_status(path)

    rec_status[str(task)] = status

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
            # dump these two which are not properties of the project per se
            del data['id']
            del data['task']
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
    return (np.stack(np.meshgrid(*arrays), axis=-1)
              .reshape(-1, ndim))

def getFrameTimestampsFromVideo(vid_file):
    """
    Parse the supplied video, return an array of frame timestamps
    """
    if vid_file.suffix in ['.mp4', '.mov']:
        # parse mp4 file
        boxes   = iso.Mp4File(str(vid_file))
        # 1. find mdat box
        moov    = boxes.child_boxes[[i for i,x in enumerate(boxes.child_boxes) if x.type=='moov'][0]]
        # 2. find track boxes
        trakIdxs= [i for i,x in enumerate(moov.child_boxes) if x.type=='trak']
        # 3. check which track contains video
        trakIdx = [i for i,x in enumerate(boxes.get_summary()['track_list']) if x['media_type']=='video'][0]
        trak    = moov.child_boxes[trakIdxs[trakIdx]]
        # 4. get mdia
        mdia    = trak.child_boxes[[i for i,x in enumerate(trak.child_boxes) if x.type=='mdia'][0]]
        # 5. get time_scale field from mdhd
        time_base = mdia.child_boxes[[i for i,x in enumerate(mdia.child_boxes) if x.type=='mdhd'][0]].box_info['timescale']
        # 6. get minf
        minf    = mdia.child_boxes[[i for i,x in enumerate(mdia.child_boxes) if x.type=='minf'][0]]
        # 7. get stbl
        stbl    = minf.child_boxes[[i for i,x in enumerate(minf.child_boxes) if x.type=='stbl'][0]]
        # 8. get sample table from stts
        samp_table = stbl.child_boxes[[i for i,x in enumerate(stbl.child_boxes) if x.type=='stts'][0]].box_info['entry_list']
        # 9. now we have all the info to determine the timestamps of each frame
        df = pd.DataFrame(samp_table) # easier to use that way
        totalFrames = df['sample_count'].sum()
        frameTs = np.zeros(totalFrames)
        # first uncompress delta table
        idx = 0
        for count,dur in zip(df['sample_count'], df['sample_delta']):
            frameTs[idx:idx+count] = dur
            idx = idx+count
        # turn into timestamps, first in time_base units
        frameTs = np.roll(frameTs,1)
        frameTs[0] = 0.
        frameTs = np.cumsum(frameTs)
        # now into timestamps in ms
        frameTs = frameTs/time_base*1000
    else:
        # open file with opencv and get timestamps of each frame
        vid = cv2.VideoCapture(str(vid_file))
        frameTs = []
        while vid.isOpened():
            # get current time (we want start time of frame
            frameTs.append(vid.get(cv2.CAP_PROP_POS_MSEC))

            # Capture frame-by-frame
            ret, frame = vid.read()

            if not ret == True:
                break

        # release the video capture object
        vid.release()
        frameTs = np.array(frameTs)

    ### convert the frame_timestamps to dataframe
    frameIdx = np.arange(0, len(frameTs))
    frameTsDf = pd.DataFrame({'frame_idx': frameIdx, 'timestamp': frameTs})
    frameTsDf.set_index('frame_idx', inplace=True)

    return frameTsDf

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

def arucoRefineDetectedMarkers(image, arucoBoard, detectedCorners, detectedIds, rejectedCorners, cameraMatrix = None, distCoeffs= None):
    corners, ids, rejectedImgPoints, recoveredIds = cv2.aruco.refineDetectedMarkers(
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
        self.vid2D = vid2D
        self.world3D = world3D
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
        # draw 3D gaze point as well, should coincide with 2D gaze point
        if self.world3D is not None and camRot is not None and camPos is not None and cameraMatrix is not None and distCoeff is not None:
            a = cv2.projectPoints(np.array(self.world3D).reshape(1,3),camRot,camPos,cameraMatrix,distCoeff)[0][0][0]
            drawOpenCVCircle(img, a, 5, (0,0,0), -1, subPixelFac)


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
    posterImageFilename= 'referencePoster.png'
    aruco_dict         = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

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
        targets = gv_config.get_targets(config_dir)
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
        markerPos = gv_config.get_markers(config_dir)
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
        return cv2.aruco.Board_create(boardCornerPoints, self.aruco_dict, np.array(ids))

    def _storeReferencePoster(self, posterImage, validationSetup):
        referenceBoard = self.getArucoBoard(unRotateMarkers = True)
        # get image with markers
        bboxExtents    = [self.bbox[2]-self.bbox[0], math.fabs(self.bbox[3]-self.bbox[1])]  # math.fabs to deal with bboxes where (-,-) is bottom left
        aspectRatio    = bboxExtents[0]/bboxExtents[1]
        refBoardWidth  = validationSetup['referencePosterWidth']
        refBoardHeight = math.ceil(refBoardWidth/aspectRatio)
        margin         = 1  # always 1 pixel, anything else behaves strangely (markers are drawn over margin as well)
        refBoardImage  = cv2.cvtColor(
            cv2.aruco.drawPlanarBoard(
                referenceBoard,(refBoardWidth+2*margin,refBoardHeight+2*margin),margin,validationSetup['markerBorderBits']),
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
    def __init__(self, ts, gaze3DRay=None, gaze3DHomography=None, lGazeOrigin=None, lGaze3D=None, rGazeOrigin=None, rGaze3D=None, gaze2DRay=None, gaze2DHomography=None, lGaze2D=None, rGaze2D=None):
        # 3D gaze is in world space, w.r.t. scene camera
        # 2D gaze is on the poster
        self.ts = ts

        # in camera space
        self.gaze3DRay        = gaze3DRay           # 3D gaze point on plane (3D gaze point <-> camera ray intersected with plane)
        self.gaze3DHomography = gaze3DHomography    # gaze2DHomography in camera space
        self.lGazeOrigin      = lGazeOrigin
        self.lGaze3D          = lGaze3D             # 3D gaze point on plane ( left eye gaze vector intersected with plane)
        self.rGazeOrigin      = rGazeOrigin
        self.rGaze3D          = rGaze3D             # 3D gaze point on plane (right eye gaze vector intersected with plane)

        # in poster space
        self.gaze2DRay        = gaze2DRay           # gaze3DRay in poster space
        self.gaze2DHomography = gaze2DHomography    # Video gaze point directly mapped to poster through homography transformation
        self.lGaze2D          = lGaze2D             # lGaze3D in poster space
        self.rGaze2D          = rGaze2D             # rGaze3D in poster space

    @staticmethod
    def getWriteHeader():
        header = ['gaze_timestamp']
        header.extend(getXYZLabels(['gazePosCam_vidPos_ray','gazePosCam_vidPos_homography']))
        header.extend(getXYZLabels(['gazeOriCamLeft','gazePosCamLeft']))
        header.extend(getXYZLabels(['gazeOriCamRight','gazePosCamRight']))
        header.extend(getXYZLabels('gazePosPoster2D_vidPos_ray',2))
        header.extend(getXYZLabels('gazePosPoster2D_vidPos_homography',2))
        header.extend(getXYZLabels('gazePosPoster2DLeft',2))
        header.extend(getXYZLabels('gazePosPoster2DRight',2))
        return header

    @staticmethod
    def getMissingWriteData():
        return [math.nan for x in range(24)]

    def getWriteData(self):
        writeData = [self.ts]
        # in camera space
        writeData.extend(allNanIfNone(self.gaze3DRay,3))
        writeData.extend(allNanIfNone(self.gaze3DHomography,3))
        writeData.extend(allNanIfNone(self.lGazeOrigin,3))
        writeData.extend(allNanIfNone(self.lGaze3D,3))
        writeData.extend(allNanIfNone(self.rGazeOrigin,3))
        writeData.extend(allNanIfNone(self.rGaze3D,3))
        # in poster space
        writeData.extend(allNanIfNone(self.gaze2DRay,2))
        writeData.extend(allNanIfNone(self.gaze2DHomography,2))
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
                lGazeOrigin     = dataReaderHelper(entry,'gazeOriCamLeft')
                lGaze3D         = dataReaderHelper(entry,'gazePosCamLeft')
                rGazeOrigin     = dataReaderHelper(entry,'gazeOriCamRight')
                rGaze3D         = dataReaderHelper(entry,'gazePosCamRight')
                gaze2DRay       = dataReaderHelper(entry,'gazePosPoster2D_vidPos_ray',2)
                gaze2DHomography= dataReaderHelper(entry,'gazePosPoster2D_vidPos_homography',2)
                lGaze2D         = dataReaderHelper(entry,'gazePosPoster2DLeft',2)
                rGaze2D         = dataReaderHelper(entry,'gazePosPoster2DRight',2)
                gaze = GazePoster(ts, gaze3DRay, gaze3DHomography, lGazeOrigin, lGaze3D, rGazeOrigin, rGaze3D, gaze2DRay, gaze2DHomography, lGaze2D, rGaze2D)

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
            reference.draw(img, self.gaze2DRay[0],self.gaze2DRay[1], subPixelFac, (0,0,0), 3)

class PosterPose:
    def __init__(self, frameIdx, nMarkers=0, rVec=None, tVec=None, hMat=None):
        self.frameIdx   = frameIdx

        # pose
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
        header = ['frame_idx','poseNMarker']
        header.extend(getXYZLabels(['poseRvec','poseTvec']))
        header.extend(['homography[%d,%d]' % (r,c) for r in range(3) for c in range(3)])
        return header

    @staticmethod
    def getMissingWriteData():
        dat = [0]
        return dat.extend([math.nan for x in range(15)])

    def getWriteData(self):
        writeData = [self.frameIdx, self.nMarkers]
        # in camera space
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
            args = tuple(noneIfAnyNan(row[c].to_numpy()) for c in (rCols,tCols,hCols))

            # insert if any non-None
            if not np.all([x is None for x in args]):   # check for not all isNone
                poses[frame_idx] = PosterPose(frame_idx,int(row['poseNMarker']),*args)

        return poses

    def setPose(self,rVec,tVec):
        self.rVec = rVec
        self.tVec = tVec

        # clear internal variables
        self._RMat        = None
        self._RtMat       = None
        self._planeNormal = None
        self._planePoint  = None
        self._RMatInv     = None
        self._RtMatInv    = None

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

        # project gaze to reference poster using camera pose
        if gaze.world3D is not None:
            # turn 3D gaze point provided by eye tracker into ray from camera
            g3D = np.matmul(RCam,np.array(gaze.world3D).reshape(3,1))
        else:
            # turn observed gaze position on video into position on tangent plane
            g3D = unprojectPoint(gaze.vid2D[0],gaze.vid2D[1],cameraMatrix,distCoeffs)

        # find intersection of 3D gaze with poster, draw
        gazePoster.gaze3DRay = posterPose.vectorIntersect(g3D)   # default vec origin (0,0,0) because we use g3D from camera's view point

        # above intersection is in camera space, turn into poster space to get position on poster
        (x,y,z)   = posterPose.camToWorld(gazePoster.gaze3DRay)  # z should be very close to zero
        gazePoster.gaze2DRay = [x, y]

    # unproject 2D gaze point on video to point on poster (should yield values very close to
    # the above method of intersecting 3D gaze point ray with poster)
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
        # get gaze vector and point on vector (pupil center) ->
        # transform from ET data coordinate frame into camera coordinate frame
        gVec    = np.matmul(RCam ,          gVec    )
        gOri    = np.matmul(RtCam,np.append(gOri,1.))
        setattr(gazePoster,attr[0],gOri)

        # intersect with poster -> yield point on poster in camera reference frame
        gPoster = posterPose.vectorIntersect(gVec, gOri)
        setattr(gazePoster,attr[1],gPoster)

        # transform intersection with poster from camera space to poster space
        if not math.isnan(gPoster[0]):
            (x,y,z)  = posterPose.camToWorld(gPoster)  # z should be very close to zero
            pgPoster = [x, y]
        else:
            pgPoster = [np.nan, np.nan]
        setattr(gazePoster,attr[2],pgPoster)

    return gazePoster

def selectDictRange(theDict,start,end):
    return {k: theDict[k] for k in theDict if k>=start and k<=end}


__all__ = ['make_video','EyeTracker','Recording']