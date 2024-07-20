import pathlib
import threading

from glassesTools import aruco, plane, recording
from glassesTools.video_gui import GUI

from .. import config
from .. import utils


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
        gui.add_window(working_dir.name)
        gui.set_show_controls(True)
        gui.set_show_play_percentage(True)

        proc_thread = threading.Thread(target=do_the_work, args=(working_dir, config_dir, gui, show_rejected_markers))
        proc_thread.start()
        gui.start()
        proc_thread.join()
    else:
        do_the_work(working_dir, config_dir, None, False)


def do_the_work(working_dir, config_dir, gui, show_rejected_markers):
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

    plane_setup = {'default': {'plane': poster, 'aruco_params': {'markerBorderBits': validationSetup['markerBorderBits']}, 'min_num_markers': validationSetup['minNumMarkers']}}

    poses, _, _ = \
        aruco.run_pose_estimation(in_video, working_dir / "frameTimestamps.tsv", working_dir / "calibration.xml",   # input video
                                  # output
                                  working_dir,
                                  # intervals to process
                                  {'default': analyzeFrames},
                                  # detector and pose estimator setup
                                  plane_setup, None,
                                  # other functions to run
                                  None,
                                  # visualization setup
                                  gui, 8, show_rejected_markers)

    plane.write_list_to_file(poses, working_dir/'posterPose.tsv', skip_failed=True)

    utils.update_recording_status(working_dir, utils.Task.Markers_Detected, utils.Status.Finished)