import pathlib

from glassesTools import fixation_classification
from glassesTools.validation import config

from .. import utils


def process(working_dir: str|pathlib.Path, config_dir: str|pathlib.Path=None):
    working_dir  = pathlib.Path(working_dir)
    if config_dir is not None:
        config_dir = pathlib.Path(config_dir)

    print('processing: {}'.format(working_dir.name))
    utils.update_recording_status(working_dir, utils.Task.Fixations_Classified, utils.Status.Running, skip_if_missing=True)

    # open file with information about Aruco marker and gaze target locations
    validationSetup = config.get_validation_setup(config_dir)

    # get interval coded to be analyzed
    analyzeFrames = utils.readMarkerIntervalsFile(working_dir / 'markerInterval.tsv')
    if analyzeFrames is None:
        print('  no marker intervals defined for this recording, skipping')
        return

    validation_plane = config.plane.ValidationPlane(config_dir, validationSetup)

    plot_limits = [[validation_plane.bbox[0]-validation_plane.marker_size, validation_plane.bbox[2]+validation_plane.marker_size],
                   [validation_plane.bbox[1]-validation_plane.marker_size, validation_plane.bbox[3]+validation_plane.marker_size]]
    fixation_classification.from_plane_gaze(working_dir/'gazePlane.tsv',
                                            analyzeFrames,
                                            working_dir,
                                            filename_stem=f'fixations',
                                            plot_limits=plot_limits)

    utils.update_recording_status(working_dir, utils.Task.Fixations_Classified, utils.Status.Finished, skip_if_missing=True)
