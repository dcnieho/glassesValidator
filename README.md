[![Downloads](https://static.pepy.tech/badge/glassesvalidator)](https://pepy.tech/project/glassesvalidator)
[![Citation Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapi.juleskreuer.eu%2Fcitation-badge.php%3Fshield%26doi%3D10.3758%2Fs13428-023-02105-5&color=blue)](https://scholar.google.com/citations?view_op=view_citation&citation_for_view=uRUYoVgAAAAJ:uWQEDVKXjbEC)
[![PyPI Latest Release](https://img.shields.io/pypi/v/glassesValidator.svg)](https://pypi.org/project/glassesValidator/)
[![image](https://img.shields.io/pypi/pyversions/glassesValidator.svg)](https://pypi.org/project/glassesValidator/)
[![DOI](https://zenodo.org/badge/DOI/10.3758/s13428-023-02105-5.svg)](https://doi.org/10.3758/s13428-023-02105-5)

# GlassesValidator v1.2.2
Tool for automatic determination of data quality (accuracy and precision) of wearable eye tracker recordings.

If you use this tool or any of the code in this repository, please cite:<br>
[Niehorster, D.C., Hessels, R.S., Benjamins, J.S., Nyström, M. and Hooge, I.T.C. (2023). GlassesValidator:
A data quality tool for eye tracking glasses. Behavior Research Methods. doi: 10.3758/s13428-023-02105-5](https://doi.org/10.3758/s13428-023-02105-5)

# How to acquire
GlassesValidator is available from `https://github.com/dcnieho/glassesValidator`, and supports Python 3.10 and 3.11 on Windows, MacOS and Linux. 

For Windows users who wish to use the glassesValidator GUI, the easiest way to acquire glassesValidator is to [download
a standalone executable](https://github.com/dcnieho/glassesValidator/releases/latest). The standalone executable is not
available for MacOS or Linux.

For users on Windows, Mac or Linux who wish to use glassesValidator in their Python code, the easiest way to acquire
glassesValidator is to install it directly into your Python distribution using the command
`python -m pip install glassesValidator`. Should that fail, this repository is pip-installable as well:
`python -m pip install git+https://github.com/dcnieho/glassesValidator.git#egg=glassesValidator`. NB: on some platforms you may have
to replace `python` with `python3` in the above command lines.

Once pip-installed in your Python distribution, there are three ways to run the GUI on any of the supported operating systems:
1. Directly in the terminal of your operating system, type `glassesValidator` and run it.
2. Open a Python console. From such a console, running the GUI requires only the following two lines of code:
    ```python
    import glassesValidator
    glassesValidator.GUI.run()
    ```
3. If you run the glassesValidator's GUI from a script, make sure to wrap your script in `if __name__=="__main__"`. This is required for correct operation from a script because the GUI uses multiprocessing functionality. Do as follows:
    ```python
    if __name__=="__main__":
        import glassesValidator
        glassesValidator.GUI.run()
    ```

# Usage
The glassesValidator validation procedure consists of two parts, 1) a poster and validation procedure that is used during a recording, and 2) Python software
for offline processing of the recording to estimate data quality measures. The glassesValidator package includes a graphical user interface (GUI)
that can be used to perform all processing. Below we describe an example workflow using the GUI. Advanced users can however opt to call all the GUI's
functionality directly from their own Python scripts without making use of the graphical user interface. The interested reader is referred to the glassesValidator
manual for further details regarding how to use the glassesValidator functionality directly from their own scripts.

## Workflow and example data
Here we first present an example workflow using the GUI. More detailed information about [using the GUI](#the-gui), or [the programming API](#api), are provided
below.

1. Before recording, the researcher prints [the poster](#the-poster) included with glassesValidator on A2 paper. See the instructions in [the section about the
   poster](#the-poster) for checking if the poster is printed correctly.
2. Before recording, the researcher hangs the printed poster on a flat surface, such as a wall. Vertical positioning of the poster depends on the experiment
   setting, but we think that a suitable default is to hang the poster such that the top row of fixation targets is at eye height for an average-length participant.
3. The operator positions the participant in front of the glassesValidator poster. An easy method for positioning the participant is to ask them to stretch their
   arm out straight forward and stand at a distance where their fist touches the poster. The operator then issues the following suggested instructions:
   `Look at the nine fixation targets in reading order for one second each. Start with the top-left (red) target. When looking at the fixation targets, keep your
   head as still as possible, move only your eyes.` These verbal instructions could be accompanied by pointing at the fixation targets in the desired looking order
   to further help the participant to follow the instructions.
4. To start calculating accuracy and precision, the researcher imports the recordings for which data quality should be determined into a glassesValidator project
   (after any [preprocessing if required](#required-preprocessing-outside-glassesvalidator), for instance by drag-dropping a folder with recordings onto the
   glassesValidator GUI window and selecting the import action.
5. Once imported, the researcher indicates which episode(s) of each recording contain a validation using a graphical interface included with glassesValidator. When performing this step, a GUI appears for each recording, playing back the scene video with gaze overlaid. To code one or multiple validation intervals, do the following:
    - The scene video will start playing automatically. You can perform action in the GUI using keypresses (make sure scene video GUI has focus so that the keypresses are received). All actions you can perform are listed by hovering your mouse over the `(?)` icon in the bottom right of the GUI.
    - Seek in the video to the start of the validation interval. To skip forward one second in the video, press `L`. For 10 seconds, press `shift+L`. To go one or ten seconds backward in the video, press `H` or `shift+H`.
    - Pause playback, and use the `J` and `K` keys to go forward and backward by one frame to precisely locate where the observer starts fixating the first validation target.
    - Press `F` to mark this frame as the start of a validation interval.
    - Seek in the video to the end of the validation interval. Pressing `P` restarts playback of the video. Once at the end of the validation interval, pause again and again use `J` and `K` to find the precise frame where the fixation on the last validation target ends.
    - Press `F` to mark this frame as the end of a validation interval.
    - If you have more validation intervals in the video, then seek further in the video to the next interval(s) and repeat the above actions to mark each.
    - Once done, press `Q` to close the GUI.
6. The recordings are then further processed automatically, and data quality is determined for validation episodes in the recording.
7. Finally, once all recordings have been processed, the researcher exports the data quality measures from the recordings in the project into a summary Excel file.
   This summary function can optionally average the data quality values over the fixation targets for each recording.

Example recordings with which steps 4-7 can be practiced are included in the [example_data subfolder](/example_data). This folder contains example recordings of a
participant executing the validation procedure with a Pupil Invisible and a Tobii Pro Glasses 2. You can import these recordings directly into glassesValidator.
Also included in each recordings subfolder is an example `analysisInterval.tsv` file for denoting where the validation interval is in the recording (step 5 above).
To use these files, after importing a recording into your [glassesValidator project](#glassesvalidator-projects) (step 4 above), copy the corresponding
`analysisInterval.tsv` to the recording's folder in the glassesValidator project before running the `Code validation intervals` action.

## glassesValidator projects
The glassesValidator GUI organizes recordings into a project folder. Each recording to be processed is imported into this project folder
and all further processing is done inside it. The source directories containing the original recordings remain untouched when running
glassesValidator. The glassesValidator project folder can furthermore contain a folder specifying the configuration of the project.
Such a configuration should be made if you used a poster different from the default (if no configuration folder is present, the default
settings are automatically used), and can be deployed with the `Deploy config` button in the GUI, or the
`glassesValidator.config.deploy_validation_config()` call from Python.

## Eye trackers
glassesValidator supports the following eye trackers:
- AdHawk MindLink
- Pupil Core
- Pupil Invisible
- Pupil Neon
- SeeTrue
- SMI ETG 1 and ETG 2
- Tobii Pro Glasses 2
- Tobii Pro Glasses 3

Pull requests or partial help implementing support for further wearable eye trackers are gladly received. To support a new eye tracker,
device support in [`glassesValidator.preprocess`](#glassesvalidatorpreprocess) should be implemented and the new eye tracker added to
the [`glassesValidator.utils.EyeTracker` `Enum`](#glassesvalidatorutils).

### Required preprocessing outside glassesValidator
For some eye trackers, the recording delivered by the eye tracker's recording unit or software can be directly imported into
glassesValidator. Recordings from some other eye trackers however require some steps to be performed in the manufacturer's
software before they can be imported into glassesValidator. These are:
- *Pupil Labs eye trackers*: Recordings should either be preprocessed using Pupil Player (*Pupil Core* and *Pupil Invisible*) or exported
  from Pupil Cloud (*Pupil Invisible* and *Pupil Neon*).
  - Using Pupil Player (*Pupil Core* and *Pupil Invisible*): Each recording should 1) be opened in Pupil Player, and 2) an export of the
    recording (`e` hotkey) should be run from pupil player. Make sure to disable the `World Video Exporter` in the `Plugin Manager`
    before exporting, as the exported video is not used by glassesValidator and takes a long time to create. Note that importing a Pupil
    Player export of a Pupil Invisible export requires an internet connection, which is used to retrieve the scene camera calibration
    from Pupil Lab's servers.
  - Using Pupil Cloud (*Pupil Invisible* and *Pupil Neon*): Export the recordings using the `Timeseries data + Scene video` action.
  - For the *Pupil Core*, for best results you may wish to do a scene camera calibration yourself, see https://docs.pupil-labs.com/core/software/pupil-capture/#camera-intrinsics-estimation.
    If you do not do so, a generic calibration will be used by Pupil Capture during data recording, by Pupil Player during data
    analysis and by glassesValidator, which may result in incorrect accuracy values.
- *SMI ETG*: For SMI ETG recordings, access to BeGaze is required and the following steps should be performed:
  - Export gaze data: `Export` -> `Legacy: Export Raw Data to File`.
    - In the `General` tab, make sure you select the following:
      - `Channel`: enable both eyes
      - `Points of Regard (POR)`: enable `Gaze position`, `Eye position`, `Gaze vector`
      - `Binocular`: enable `Gaze position`
      - `Misc Data`: enable `Frame counter`
      - disable everything else
    - In the Details tab, set:
      - `Decimal places` to 4
      - `Decimal separator` to `point`
      - `Separator` to `Tab`
      - enable `Single file output`
    - This will create a text file with a name like `<experiment name>_<participant name>_<number> Samples.txt`
      (e.g. `005-[5b82a133-6901-4e46-90bc-2a5e6f6c6ea9]_005_001 Samples.txt`). Move this file/these files to the
      recordings folder and rename them. If, for instance, the folder contains the files `005-2-recording.avi`,
      `005-2-recording.idf` and `005-2-recording.wav`, amongst others, for the recording you want to process,
      rename the exported samples text file to `005-2-recording.txt`.
  - Export the scene video:
    - On the Dashboard, double click the scene video of the recording you want to export to open it in the scanpath tool.
    - Right click on the video and select settings. Make the following settings in the `Cursor` tab:
      - set `Gaze cursor` to `translucent dot`
      - set `Line width` to 1
      - set `Size` to 1
    - Then export the video, `Export` -> `Export Scan Path Video`. In the options dialogue, make the following settings:
      - set `Video Size` to the maximum (e.g. `(1280,960)` in my case)
      - set `Frames per second` to the framerate of the scene camera (24 in my case)
      - set `Encoder` to `Performance [FFmpeg]`
      - set `Quality` to `High`
      - set `Playback speed` to `100%`
      - disable `Apply watermark`
      - enable `Export stimulus audio`
      - finally, click `Save as`, navigate to the folder containing the recording, and name it in the same format as the
        gaze data export file we created above but replacing `recording` with `export`, e.g. `005-2-export.avi`.

## The poster
The default poster is available 1) [here](/src/glassesValidator/config/poster/poster.pdf), 2) from the GUI with the `Get poster
pdf` button, and 3) can also be acquired from a Python script by calling
`glassesValidator.config.poster.deploy_default_pdf()`.
The default poster should be printed at A2 size, as defined
in the pdf file, and is designed to cover a reasonable field of view when participants view it at armslength (i.e., 20 x 17.5 deg
at 60 cm). In order to check that the poster was printed at the correct scale, one should measure the sides of the ArUco markers.
We strongly recommend performing this check because printers may not be calibrated. In the case of the default glassesValidator
poster, the distance between the left side of the left-most column of ArUco markers and the right side of the right-most column of
ArUco markers should be 35.6 cm (each ArUco marker should have sides that are 4.19 cm long). If the poster was printed at the wrong
scale, one must adapt the glassesValidator configuration to match the size and position of the ArUco markers and fixation targets
on your poster ([see "Customizing the poster" below](#customizing-the-poster)).

### Customizing the poster
The poster pdf file is generated by the [LaTeX file in the same folder as the pdf](/src/glassesValidator/config/poster/poster.tex).
Its looks are defined in the files in the [config folder](/src/glassesValidator/config). As described above, this configuration can be
deployed and then edited. The edited configuration can both be used to generate a new poster with LaTeX and for performing the
data quality calculations.
Specifically, the files [`markerPositions.csv`](/src/glassesValidator/config/markerPositions.csv) and
[`targetPositions.csv`](/src/glassesValidator/config/targetPositions.csv) define where the ArUco markers and fixation targets
(respectively) are placed on the poster. Each coordinate in these files is for the center of the marker or gaze target and the origin
(0,0) is in the bottom left of the poster. The [`validationSetup.txt` configuration file](/src/glassesValidator/config/validationSetup.txt)
contains the following settings for the poster:

|setting|default<br>value|description|
| --- | --- | --- |
|`distance`|60|Viewing distance in cm, used to convert coordinates and sizes in degrees to cm. Only used when `mode` is `deg`.|
|`mode`|cm|`cm` or `deg`. Sets the unit for the `markerSide` and `targetDiameter` below as well as for interpreting the coordinates in the marker and target position files.|
|`markerSide`|4.18945|Size of ArUco markers. In cm or deg, see `mode` setting.|
|`markerPosFile`|[`markerPositions.csv`](/src/glassesValidator/config/markerPositions.csv)|File in the config folder where the markers to draw are specified.|
|`targetPosFile`|[`targetPositions.csv`](/src/glassesValidator/config/targetPositions.csv)|File in the config folder where the targets to draw are specified.|
|`targetType`|Thaler|Type of targer to draw, can be `Tobii` (the calibration marker used by Tobii) or `Thaler` (layout ABC in the lower panel of Fig. 1 in [Thaler et al., 2013](https://doi.org/10.1016/j.visres.2012.10.012)).|
|`targetDiameter`|1.04736|Diameter of the gaze target. In cm or deg, see `mode` setting. Ignored if `targetType` is `Tobii` and `useExactTobiiSize` is `1`.|
|`useExactTobiiSize`|0|`0` or `1`. If `1`, the gaze targets have the exact dimensions of a Tobii calibration marker (though possibly different colors). I.e., the `targetDiameter` parameter is ignored. Only used if `targetType` is `Tobii`.|
|`showGrid`|0|`0` or `1`. If `1`, a `gridCols` x `gridRows` grid is drawn behind the markers and gaze targets. The size of each grid cell is 1 cm if `mode` is cm, or 1 degree if `mode` is `deg`.|
|`gridCols`|35.6102|Number of grid columns to draw if `showGrid` is `1`.|
|`gridRows`|30.3734|Number of grid rows to draw if `showGrid` is `1`.|
|`showAnnotations`|0|`0` or `1`. If `1`, text annotations informing about the size of grid cells and markers is printed on the poster below the marker arrangement.|
|`markerBorderBits`|1|Setting for border thickness of ArUco markers. Used by the glassesValidator tool when deploying a configuration and `glassesValidator.config.poster.deploy_marker_images()` function for generating the marker images that are placed on the poster by the [poster generation LaTeX file](/src/glassesValidator/config/poster/poster.tex). Also used when detecting ArUco markers during recording processing.|

To check your custom configuration, you can generate a poster pdf using [the steps below](#steps-for-making-your-own-poster). Furthermore,
a png image showing the poster will be generated in the configuration folder when any of glassesValidator's processing steps are run.

The above settings are furthermore used by glassesValidator when processing recordings. For instance, the `distance` parameter is
used as the assumed viewing distance when computing the `viewpos_vidpos_homography` data quality type (see [the discussion in the
Advanced settings section below](#advanced-settings)). Three further settings are present in the [`validationSetup.txt` configuration
file](/src/glassesValidator/config/validationSetup.txt) that are only used by the glassesValidator processing tool, and not for the poster:

|setting|default<br>value|description|
| --- | --- | --- |
|`minNumMarkers`|3|During recording processing, minimum number of detected markers required to perform estimation of homography transformation and camera pose estimation. |
|`centerTarget`|5| The ID of the fixation target (in the [`targetPosFile`](/src/glassesValidator/config/targetPositions.csv) file) that is the origin of the poster. The center of this marker will be (0,0) in the poster coordinate system. |
|`referencePosterWidth`|1920| Width (in pixels) of the poster png image generated by the glassesValidator tool and stored in the config directory when loading a configuration.|


### Steps for making your own poster
1. Deploy the default configuration using the `Deploy config` button in the GUI, or the
   `glassesValidator.config.deploy_validation_config()` call from Python.
2. Edit the `validationSetup.txt` configuration file and the `markerPositions.csv` and `targetPositions.csv` files in the
   configuration folder to design the layout and look of the poster that you want. NB: if you edit the `markerBorderBits` 
   variable, you have to run the `glassesValidator.config.poster.deploy_marker_images()` to regenerate the marker images
   contained in the `all-markers` with the new border setting.
3. Compile the `markerBoard/board.tex` LaTeX file with `pdfTex`, such as provided in the [TeX Live
   distribution](https://www.tug.org/texlive/).
4. Done, you should now have a pdf file with the poster as you defined.

## The GUI
![Glasses viewer screenshot](/.github/images/screenshot.png?raw=true)
Click the screenshot to see a full-size version.

The simplest way to use glassesValidator is by means of its GUI, see above. The full workflow can be performed in the GUI.
Specifically, it supports:

- GlassesValidator project management, such as making new projects or opening them.
- Importing recordings into a glassesValidator project. Recordings can be found by glassesValidator by drag-dropping one or multiple
  directories onto the GUI or by clicking the `Add recordings` button (not shown) and selecting one or multiple directories. The
  selected directories and all their subdirectories are then searched for recordings, and the user can select which of the found
  recordings should be added to the glassesValidator import. Once listed in the GUI, the import action can then be started, which
  copies over data from the selected recording to the glassesValidator project folder, and transforms it to a common format.
- Showing a listing of the recordings in the project and several properties about them, where available. these properties currently
  are:
  - Eye tracker type (e.g. Pupil Invisible or Tobii Glasses 2)
  - Status, indicating whether a recording has been imported, coded, analyzed)
  - Recording name
  - Participant name
  - Project name
  - Recording duration
  - Recording start time
  - Working directory for the recording inside the glassesValidator project
  - Original source directory from which the recording was imported
  - Firmware version
  - Glasses serial number
  - Recording unit serial number
  - Recording software version
  - Scene camera serial number

  The GUI can be configured to show any combination of these columns in any order, and the listing can be sorted by any of these
  columns. The listing can furthermore be filtered to show only a subset of recordings by search string, eye tracker type and
  status.

- Annotating recordings to indicate the episode(s) in the recording that contain a validation using a separate GUI.
- Calculating data quality (accuracy and precision) for recordings.
- Exporting the data quality values of multiple recordings to a summary tab-separated spreadsheet file.
- Deploying the default configuration to a glassesValidator project so that it can be edited ([see "Customizing the poster"
  above](#customizing-the-poster)).

### Advanced settings
Several advanced settings can be made in the right bar of the glassesValidator GUI. For standard use, these are not of interest.
Each setting is explained by means of a help text that pops up when hovering over the setting. Nonetheless, the specific group
of advanced settings for configuring what type of data quality to compute is discussed here in more detail.

glassesValidator can compute data quality in multiple ways. When deciding how to determine accuracy and precision, several
decisions have to be made. By default the most appropriate decisions for standard use are selected and most users can skip this
section, but another configuration might be more suited for some advanced use cases. The following decisions are made:

1. Determining location of the participant:

   1. The researcher can assume a fixed viewing distance and provide this to glassesValidator in the project configuration. In 
      this mode, it is assumed that the eye is located exactly in front of the center of the poster and that the poster is
      oriented perpendicularly to the line of sight from this assumed viewing position.
   2. If the scene camera is calibrated (i.e. its properties such as focal length and distortion parameters have been estimated
      by a calibration procedure), it is possible to use the array of ArUco markers to estimate the position of the participant
      relative to the poster at each time point during a validation.

   When a scene camera calibration is available, glassesValidator will by default use mode ii, otherwise mode i will be used.
   Five of the six wearable eye trackers supported by glassesValidator provide the calibration of their scene camera, and
   glassesValidator will by default use this calibration for these eye trackers (mode ii). Currently, only the SeeTrue does not
   provide a camera calibration, and glassesValidator therefore by default uses an assumed fixed viewing distance for this eye
   tracker.

2) Transforming gaze positions from the scene camera reference frame to positions on the validation poster:

   1. Performed by means of homography.
   2. Performed using recovered camera pose and gaze direction vector, by means of intersection of gaze vector with the validation
      poster plane.

   Mode ii is used by default. However, like for decision 1, mode ii requires that a camera calibration is available. If a camera
   calibration is not available, mode i will be used instead.

3) Which data is used for determining gaze position on the validation poster:

   1. The gaze position in the scene camera image.
   2. The gaze position in the world (often binocular gaze point).
   3. Gaze direction vectors in a head reference frame.

   When operating in mode i, the eye tracker's estimate of the (binocular) gaze point in the scene camera image is used. This is
   the appropriate choice for most wearable eye tracking research, as it is this gaze point that is normally used for further
   analysis. However, in some settings and when the eye tracker provides a (3D) gaze position in the world and/or gaze direction
   vectors for the individual eyes along with their origin, a different mode of operation may be more appropriate. Specifically,
   when using the wearable eye tracker's world gaze point or gaze vectors instead of the gaze point in the scene video in their
   analysis, the researcher should compute the accuracy and precision of this world gaze point/gaze vectors. NB: for most of the
   currently supported eye trackers, modes i and ii are equivalent (i.e., the gaze position in the camera image is simply the
   gaze position in the world projected to the camera image). This is however not always the case. The AdHawk MindLink for instance
   has an operating mode that corrects for parallax error in the projected gaze point using the vergence signal, which leads to
   the eye tracker reporting a different gaze position in the scene video than a direct projection of gaze position in the world
   to the scene camera image.

Altogether, combining these decisions, the following seven types of data quality can be calculated using glassesValidator. NB: All API
names are members of the `enum.Enum` `glassesValidator.process.DataQualityType`

|Name in GUI|Name in API|description|
| --- | --- | --- |
|Homography + view distance|`viewpos_vidpos_homography`|Use a homography tranformation to map gaze position from the scene video to the validation poster, and use an assumed viewing distance (see the project's configuration) to compute data quality measures in degrees with respect to the scene camera. In this mode, it is assumed that the eye is located exactly in front of the center of the poster and that the poster is oriented perpendicularly to the line of sight from this assumed viewing position. *Default mode when no camera calibration is available.*|
|Homography + pose|`pose_vidpos_homography`|Use a homography tranformation to map gaze position from the scene video to the validation poster, and use the determined pose of the scene camera (requires a calibrated camera) to compute data quality measures in degrees with respect to the scene camera.|
|Video ray + pose|`pose_vidpos_ray`|Use camera calibration to turn gaze position from the scene video into a direction vector, and determine gaze position on the validation poster by intersecting this vector with the poster. Then, use the determined pose of the scene camera (requires a calibrated camera) to compute data quality measures in degrees. If the eye tracker provides a 3D gaze point (e.g. the Tobii Pro Glasses 2 and 3), this 3D gaze point is used in lieu of a gaze direction vector derived from the gaze position in the scene camera with respect to the scene camera. *Default mode when a camera calibration is available.*|
|World gaze position + pose|`pose_world_eye`|Use the gaze position in the world provided by the eye tracker (often a binocular gaze point) to determine gaze position on the validation poster by turning it into a direction vector with respect to the scene camera and intersecting this vector with the poster. Then, use the determined pose of the scene camera (requires a calibrated camera) to compute data quality measures in degrees with respect to the scene camera.|
|Left eye ray + pose|`pose_left_eye`|Use the gaze direction vector for the left eye provided by the eye tracker to determine gaze position on the validation poster by intersecting this vector with the poster. Then, use the determined pose of the scene camera (requires a camera calibration) to compute data quality measures in degrees with respect to the left eye.|
|Right eye ray + pose|`pose_right_eye`|Use the gaze direction vector for the right eye provided by the eye tracker to determine gaze position on the validation poster by intersecting this vector with the poster. Then, use the determined pose of the scene camera (requires a camera calibration) to compute data quality measures in degrees with respect to the right eye.|
|Average eye rays + pose|`pose_left_right_avg`|For each time point, take angular offset between the left and right gaze positions and the fixation target and average them to compute data quality measures in degrees. Requires 'Left eye ray + pose' and 'Right eye ray + pose' to be enabled|

In summary, _by default_ for eye trackers for which a camera calibration is available the gaze position in the scene camera is
transformed to a gaze position on the validation poster by means of intersecting a camera-relative gaze direction vector with the
validation poster plane. Accuracy and precision are then computed using the angle between the vectors from the scene camera to the
fixation target and to the gaze position on the poster (`glassesValidator.process.DataQualityType.pose_vidpos_ray`). For eye trackers
for which no camera calibration is available, a homography transformation is used to determine gaze position on the poster and accuracy
and precision are computed using an assumed viewing distance configured in the glassesValidator project's configuration file, along
with the assumptions that the eye is located exactly in front of the center of the poster and that the poster is oriented perpendicularly
to the line of sight (`glassesValidator.process.DataQualityType.viewpos_vidpos_homography`). As discussed in the "Assuming a fixed
viewing distance" section of the glassesValidator paper (Niehorster et al., 2023), differences in computed values between these two
modes are generally small. Nonetheless, it is up to the researcher to decide whether the level of error introduced when operating
without a camera calibration is acceptable and whether they should perform their own camera calibration.

### Matching gaze data to fixation targets
This section discusses how to decide which part of the gaze data constitutes a fixation on each of the fixation targets.
By default, to associate gaze data with fixation targets, glassesValidator first uses the [I2MC fixation
classifier](https://github.com/dcnieho/I2MC_Python) to classify the fixations in the gaze position data on the poster.
Then, for each fixation target, the nearest fixation in poster-space that is at least 50 ms long is selected from all the classified
fixations during the validation procedure. This matching is done in such a way that no fixation is matched to more than one fixation
target.

The matching between fixations and fixation targets produced by this procedure is stored in a file `analysisInterval.tsv` in each
recording's directory in the glassesValidator project. The advanced user can provide their own matching by changing the contents of
this file.

### Coordinate system of data
Gaze data in poster space in the `gazePosterPos.tsv` file of a processed recording has its origin (0,0) at the center of the position
of the fixation target that was indicated to be the center target with the `centerTarget` setting in the [`validationSetup.txt`
configuration file](/src/glassesValidator/config/validationSetup.txt). The positive x-axis points to the right and the positive y-axis
downward, which means that (-,-) coordinates are to the left and above of the poster origin, and (+,+) to the right and below.

Angular accuracy values in the `dataQuality.tsv` file of a processed recording use the same sign-coding as the gaze data in poster space.
That is, for the horizontal component of reported accuracy values, positive means gaze is to the right of the fixation target and
negative to the left. For the vertical component, positive means gaze is below the fixation target, and negative that it is above the
fixation target.

## API
All of glassesValidator's functionality is exposed through its API. Below are all functions that are part of the
public API. Many functions share common input arguments. These are documented [here](#common-input-arguments) and linked to in the API
overview below.

### glassesValidator.config
|function|inputs|description|
| --- | --- | --- |
|`get_validation_setup()`|<ol><li>[`config_dir`](#common-input-arguments)</li><li>`setup_file`: filename of the validation setup file. Default `validationSetup.txt`.</li></ol>|Read and parse the validation setup into a dict. If no `config_dir` is provided, the default configuration is returned.|
|`get_targets()`|<ol><li>[`config_dir`](#common-input-arguments)</li><li>`file`: filename of the target positions file. Default `targetPositions.csv`.</li></ol>|Read and parse the target positions file into a `pandas.DataFrame`. If no `config_dir` is provided, the target positions are returned.|
|`get_markers()`|<ol><li>[`config_dir`](#common-input-arguments)</li><li>`file`: filename of the marker positions file. Default `markerPositions.csv`.</li></ol>|Read and parse the marker positions file into a `pandas.DataFrame`. If no `config_dir` is provided, the marker positions are returned.|
|`deploy_validation_config()`|<ol><li>`output_dir`: Directory to deploy configuration files to.</li></ol>|Deploy the default configuration (including poster maker, see `glassesValidator.config.poster.deploy_maker()`) to the specified directory.|

#### glassesValidator.config.poster
|function|inputs|description|
| --- | --- | --- |
|`deploy_maker()`|<ol><li>`output_dir`: Directory to deploy poster maker LaTeX files to.</li></ol>|Deploy the LaTeX file for making posters (including marker images, see `glassesValidator.config.poster.deploy_marker_images()`) to the specified directory.|
|`deploy_marker_images()`|<ol><li>`output_dir`: Directory to deploy marker images for use with the LaTeX poster maker to.</li></ol>|Deploy the marker images (all fiducial markers in `cv2.aruco.DICT_4X4_250`) to the specified directory.|
|`deploy_default_pdf()`|<ol><li>`output_file_or_dir`: Path where to store default poster pdf. If provided path is a directory and not a file, the poster is stored as `poster.pdf` in the provided directory.</li></ol>|Deploy the default poster pdf to the specified path.|

### glassesValidator.GUI
|function|inputs|description|
| --- | --- | --- |
|`run()`|<ol><li>`project_dir`: Path to glassesValidator project to open.</li></ol>|Open the glassesValidator GUI.|

### glassesValidator.preprocess
|function|inputs|description|
| --- | --- | --- |
|`get_recording_info()`|<ol><li>[`source_dir`](#common-input-arguments)</li><li>`device`: `glassesValidator.utils.EyeTracker`</li></ol>|Determine if provided path contains a recording/recordings made with the specified eye tracker (`device`) and if so, get info about these recordings.|
|`do_import()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>`device`: `glassesValidator.utils.EyeTracker`</li><li>[`rec_info`](#common-input-arguments)</li></ol>|Import the specified recording to a subdirectory of `output_dir`. Either `device` or `rec_info` must be specified. Does nothing if directory does not contain a recording made with the specified eye tracker.|
|  |  |  |
|`adhawk_mindlink()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>[`rec_info`](#common-input-arguments)</li></ol>|Import an AdHawk MindLink recording to a subdirectory of `output_dir`. Does nothing if directory does not contain an AdHawk MindLink recording. `rec_info` is optional.|
|`pupil_core()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>[`rec_info`](#common-input-arguments)</li></ol>|Import a Pupil Core recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a Pupil Core recording. `rec_info` is optional.|
|`pupil_invisible()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>[`rec_info`](#common-input-arguments)</li></ol>|Import a Pupil Invisible recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a Pupil Invisible recording. `rec_info` is optional.|
|`pupil_neon()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>[`rec_info`](#common-input-arguments)</li></ol>|Import a Pupil Neon recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a Pupil Neon recording. `rec_info` is optional.|
|`SeeTrue()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>[`rec_info`](#common-input-arguments)</li><li>`cam_cal_file`: OpenCV XML file containing a camera calibration to be used when processing this recording. Optional.</li></ol>|Import a SeeTrue recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a SeeTrue recording. `rec_info` is optional.|
|`SMI_ETG()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>[`rec_info`](#common-input-arguments)</li></ol>|Import a SMI ETG recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a SMI ETG 1 or 2 recording. `rec_info` is optional.|
|`tobii_G2()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>[`rec_info`](#common-input-arguments)</li></ol>|Import a Tobii Pro Glasses 2 recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a Tobii Pro Glasses 2 recording. `rec_info` is optional.|
|`tobii_G3()`|<ol><li>[`output_dir`](#common-input-arguments)</li><li>[`source_dir`](#common-input-arguments)</li><li>[`rec_info`](#common-input-arguments)</li></ol>|Import a Tobii Pro Glasses 3 recording to a subdirectory of `output_dir`. Does nothing if directory does not contain a Tobii Pro Glasses 3 recording. `rec_info` is optional.|

### glassesValidator.process
|function|inputs/members|description|
| --- | --- | --- |
|`do_coding()`|<ol><li>[`working_dir`](#common-input-arguments)</li><li>[`config_dir`](#common-input-arguments)</li></ol>|Show GUI for indicating which episode(s) in the recording contain a validation. Simplified alias of `glassesValidator.process.code_marker_interval()`.|
|`do_process()`|<ol><li>[`working_dir`](#common-input-arguments)</li><li>[`config_dir`](#common-input-arguments)</li></ol>|Perform all steps to compute data quality for a recording for which the validation episodes have been coded. Use the below functions instead for granular control over which steps are run.|
|`export_data_quality`|<ol><li>`rec_dirs`: list of recording directories (glassesValidator [`working_dir`](#common-input-arguments)s) that contain a data quality file and are to be included in the summary file.</li><li>`output_file_or_dir`: Path where to store the summary file. If provided path is a directory and not a file, the summary file is stored as `dataQuality.tsv` in the provided directory.</li><li>`dq_types`: list of `glassesValidator.process.DataQualityType`s to include in the export (if available). If not provided, a suitable default is chosen.</li><li>`targets`: list of targets to include in the export (and optionally average over). If not provided, all targets are included.</li><li>`average_over_targets`: if `True`, all measured are averaged over the targets listed in `targets`, providing one value per validation episode.</li><li>`include_data_loss`: if `True`, data loss is also included among the computed data quality metrics (if available). **Note however** that this data loss is computed during the episode selected for each fixation target on the validation poster. This is **NOT** the data loss of the whole recording and thus not what you want to report in your paper.</li></ol>|Export the data quality values of multiple recordings to a summary tab-separated spreadsheet file.|
|  |  |  |
|`code_marker_interval()`|<ol><li>[`working_dir`](#common-input-arguments)</li><li>[`config_dir`](#common-input-arguments)</li><li>`show_poster`: if `True`, also show poster with gaze overlaid on it in a second viewer.</li></ol>|Show GUI for indicating which episode(s) of each recording contain a validation.|
|`detect_markers()`|<ol><li>[`working_dir`](#common-input-arguments)</li><li>[`config_dir`](#common-input-arguments)</li><li>`show_visualization`: if `True`, each frame is shown in a viewer, overlaid with info about detected markers and poster.</li><li>`show_rejected_markers`: if `True`, rejected ArUco marker candidates are also shown in the viewer.</li></ol>|Detect ArUco markers on the poster in the scene video and determine camera pose with respect to the scene camera (if scene camera is calibrated) and homography to transform between scene camera and poster coordinates.|
|`gaze_to_poster()`|<ol><li>[`working_dir`](#common-input-arguments)</li><li>[`config_dir`](#common-input-arguments)</li><li>`show_visualization`: if `True`, each frame is shown in a viewer, overlaid with info about detected markers and poster.</li><li>`show_poster`: if `True`, gaze in poster space is also shown in a separate viewer.</li><li>`show_only_intervals`: if `True`, only the coded validation episodes (if available) are shown in the viewer while the rest of the scene video is skipped past.</li></ol>|Transform gaze data in scene camera reference frame to positions on the poster.|
|`compute_offsets_to_targets()`|<ol><li>[`working_dir`](#common-input-arguments)</li><li>[`config_dir`](#common-input-arguments)</li></ol>|Compute offsets from each gaze sample to each target.|
|`determine_fixation_intervals()`|<ol><li>[`working_dir`](#common-input-arguments)</li><li>`do_global_shift`: If `True`, for each validation interval the mean position will be removed from the gaze data and the targets, removing any overall shift of the data. This improves the matching of fixations to targets when there is a significant overall offset in the data. It may fail (backfire) if there are data samples far outside the range of the validation targets, or if there is no data for some targets.</li><li>`max_dist_fac`: Factor for determining distance limit when assigning fixation points to validation targets. If for a given target the closest fixation point is further away than `<factor>*[minimum intertarget distance]`, then no fixation point will be assigned to this target, i.e., it will not be matched to any fixation point. Set to a large value to essentially disable.</li><li>[`config_dir`](#common-input-arguments)</li></ol>|Automatically determine using [I2MC](https://github.com/dcnieho/I2MC_Python) which episodes during the validation constitute looks to each fixation target.|
|`calculate_data_quality()`|<ol><li>[`working_dir`](#common-input-arguments)</li><li>`dq_types`: `list` of `glassesValidator.process.DataQualityType`s to compute.</li><li>`allow_dq_fallback`: if `True`, fall back to most suitable data quality type if none of the types requested in `dq_types` are available. If False, a `RuntimeError` error is raised instead.</li><li>`include_data_loss`: if `True`, data loss is also included among the computed data quality metrics. **Note however** that this data loss is computed during the episode selected for each fixation target on the validation poster. This is **NOT** the data loss of the whole recording and thus not what you want to report in your paper.</li></ol>|Compute data quality for all fixation targets.|
|  |  |  |
|`DataQualityType`|`enum.Enum`<ul><li>`DataQualityType.viewpos_vidpos_homography`</li><li>`DataQualityType.pose_vidpos_homography`</li><li>`DataQualityType.pose_vidpos_ray`</li><li>`DataQualityType.pose_world_eye`</li><li>`DataQualityType.pose_left_eye`</li><li>`DataQualityType.pose_right_eye`</li><li>`DataQualityType.pose_left_right_avg`</li></ul>|See [Advanced settings section in The GUI section](#advanced-settings) above.|

### glassesValidator.utils
|function|inputs/members|description|
| --- | --- | --- |
|`make_video()`|<ol><li>[`working_dir`](#common-input-arguments)</li><li>[`config_dir`](#common-input-arguments)</li><li>`show_rejected_markers`: if `True`, rejected ArUco marker candidates are also drawn on the video.</li><li>`add_audio_to_poster_video`: if `True`, audio is added to poster video, not only to the scene video.</li><li>`show_visualization`: if `True`, the generated video is shown as it is created in a viewer.</li></ol>|Export annotated scene video showing gaze, detected ArUco markers and detected poster. Also exports a video showing gaze on the poster.|
|  |  |  |
|`EyeTracker`|`enum.Enum`<ul><li>`EyeTracker.Pupil_Core`</li><li>`EyeTracker.Pupil_Invisible`</li><li>`EyeTracker.Pupil_Neon`</li><li>`EyeTracker.SMI_ETG`</li><li>`EyeTracker.SeeTrue`</li><li>`EyeTracker.Tobii_Glasses_2`</li><li>`EyeTracker.Tobii_Glasses_3`</li></ul>||
|`Recording`|`dataclasses.dataclass`<ul><li>`Recording.name`</li><li>`Recording.source_directory`</li><li>`Recording.proc_directory_name`</li><li>`Recording.start_time`</li><li>`Recording.duration`</li><li>`Recording.eye_tracker`: `glassesValidator.utils.EyeTracker`</li><li>`Recording.project`</li><li>`Recording.participant`</li><li>`Recording.firmware_version`</li><li>`Recording.glasses_serial`</li><li>`Recording.recording_unit_serial`</li><li>`Recording.recording_software_version`</li><li>`Recording.scene_camera_serial`</li><li>`Recording.id` (internal to GUI)</li><li>`Recording.task` (internal to GUI)</li></ul>|Information about a recording|

### Common input arguments
|argument|module(s)|description|
| --- | --- | --- |
|`config_dir`|`glassesValidator.config`<br>`glassesValidator.process`<br>`glassesValidator.utils`|Path to directory containing a glassesValidator setup. If `None`, the default setup is used.|
|`source_dir`|`glassesValidator.preprocess`|Path to directory containing one (or for some eye trackers potentially multiple) eye tracker recording(s) as stored by the eye tracker's recording hardware or software.|
|`output_dir`|`glassesValidator.preprocess`|Path to the directory to which recordings will be imported. Each recording will be placed in a subdirectory of the specified path.|
|`working_dir`|`glassesValidator.process`<br>`glassesValidator.utils`|Path to a glassesValidator recording directory.|
|`rec_info`|`glassesValidator.preprocess`|Recording info (`glassesValidator.utils.Recording`) or list of recording info specifying what is expected to be found in the specified `source_dir`, so that this does not have to be rediscovered and changes can be made e.g. to the recording name that is used for auto-generating the recording's `working_dir`, or even directly specifying the `working_dir` by filling the `proc_directory_name` field before import.|
