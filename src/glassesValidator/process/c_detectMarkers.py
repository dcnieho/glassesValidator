import pathlib
import threading

from glassesTools import aruco, recording

from .. import config
from .. import utils
from ._image_gui import GUI, generic_tooltip, qns_tooltip


stopAllProcessing = False
def process(working_dir, config_dir=None, show_visualization=False, show_rejected_markers=False):
    # if show_visualization, each frame is shown in a viewer, overlaid with info about detected markers and poster
    # if show_rejected_markers, rejected ArUco marker candidates are also shown in the viewer. Possibly useful for debug
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))

    # if we need gui, we run processing in a separate thread (GUI needs to be on the main thread for OSX, see https://github.com/pthom/hello_imgui/issues/33)
    if show_visualization:
        gui = GUI(use_thread = False)
        gui.set_interesting_keys('qns')
        gui.register_draw_callback('status',lambda: generic_tooltip(qns_tooltip()))
        gui.add_window(working_dir.name)

        proc_thread = threading.Thread(target=do_the_work, args=(working_dir, config_dir, gui, show_rejected_markers))
        proc_thread.start()
        gui.start()
        proc_thread.join()
        return stopAllProcessing
    else:
        return do_the_work(working_dir, config_dir, None, False)


def do_the_work(working_dir, config_dir, gui, show_rejected_markers):
    global stopAllProcessing

    utils.update_recording_status(working_dir, utils.Task.Markers_Detected, utils.Status.Running)

    # get info about recording
    recInfo = recording.Recording.load_from_json(working_dir)

    # open file with information about Aruco marker and Gaze target locations
    validationSetup = config.get_validation_setup(config_dir)
    # get info about markers on our poster
    poster          = config.poster.Poster(config_dir, validationSetup)

    # get interval(s) coded to be analyzed, if any
    analyzeFrames   = utils.readMarkerIntervalsFile(working_dir / "markerInterval.tsv")

    # open video file, query it for size
    in_video = recInfo.get_scene_video_path()

    stopAllProcessing = aruco.run_pose_estimation(in_video, working_dir / "frameTimestamps.tsv", working_dir / "calibration.xml",   # input video
                                                  # output
                                                  working_dir, 'posterPose.tsv',
                                                  # intervals to process
                                                  analyzeFrames,
                                                  # detector and pose estimator setup
                                                  poster.getArucoBoard(), {'markerBorderBits': validationSetup['markerBorderBits']}, validationSetup['minNumMarkers'],
                                                  # visualization setup
                                                  gui, poster.markerSize/2, 8, show_rejected_markers)

    utils.update_recording_status(working_dir, utils.Task.Markers_Detected, utils.Status.Finished)

    return stopAllProcessing