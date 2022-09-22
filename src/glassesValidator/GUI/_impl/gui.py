from imgui.integrations.glfw import GlfwRenderer
import concurrent.futures
from PIL import Image
import configparser
import platform
import asyncio
import pathlib
import OpenGL
import OpenGL.GL as gl
import glfw
import imgui
import time
import sys
import datetime
import typing
import io
from importlib.resources import files, as_file


from .structs import DefaultStyleDark, DefaultStyleLight, Filter, FilterMode, MsgBox, Os, SortSpec, filter_mode_names
from . import globals, async_thread, callbacks, db, filepicker, imagehelper, msgbox, utils
from ...utils import EyeTracker, Recording, Status, hex_to_rgba_0_1, eye_tracker_names, status_names

imgui.io = None
imgui.style = None

def draw_tooltip(hover_text):
    imgui.begin_tooltip()
    imgui.push_text_wrap_pos(min(imgui.get_font_size() * 35, imgui.io.display_size.x))
    imgui.text_unformatted(hover_text)
    imgui.pop_text_wrap_pos()
    imgui.end_tooltip()

def draw_hover_text(hover_text: str, text="(?)", force=False, *args, **kwargs):
    if text:
        imgui.text_disabled(text, *args, **kwargs)
    if force or imgui.is_item_hovered():
        draw_tooltip(hover_text)
        return True
    return False


class RecordingTable():
    def __init__(self,
                 recordings: dict[int, Recording],
        selected_recordings: dict[int, bool],
        in_adder_popup: bool = False):
        
        self.recordings = recordings
        self.selected_recordings = selected_recordings
        self.in_adder_popup = in_adder_popup

        self.sorted_recordings_ids: list[int] = []
        self.require_sort: bool = True
        self.filters: list[Filter] = []
        self.filter_box_text: str = ""

        self._recording_list_column_count = 15
        self._num_recordings = len(self.recordings)
        self._eye_tracker_label_width: float = None
        self.table_flags: int = (
            imgui.TABLE_SCROLL_X |
            imgui.TABLE_SCROLL_Y |
            imgui.TABLE_HIDEABLE |
            imgui.TABLE_SORTABLE |
            imgui.TABLE_RESIZABLE |
            imgui.TABLE_SORT_MULTI |
            imgui.TABLE_REORDERABLE |
            imgui.TABLE_ROW_BACKGROUND |
            imgui.TABLE_SIZING_FIXED_FIT |
            imgui.TABLE_NO_HOST_EXTEND_Y |
            imgui.TABLE_NO_BORDERS_IN_BODY_UTIL_RESIZE
        )

    def add_filter(self, filter):
        self.filters.append(filter)
        self.require_sort = True

    def remove_filter(self, id):
        for i, search in enumerate(self.filters):
            if search.id == id:
                self.filters.pop(i)
        self.require_sort = True

    def font_changed(self):
        self._eye_tracker_label_width = None

    def draw(self):
        extra = "_adder" if self.in_adder_popup else ""
        if imgui.begin_table(
            f"###recording_list{extra}",
            column=self._recording_list_column_count,
            flags=self.table_flags,
        ):
            if (num_recordings := len(self.recordings)) != self._num_recordings:
                self._num_recordings = num_recordings
                self.require_sort = True
            frame_height = imgui.get_frame_height()

            # Setup
            checkbox_width = frame_height
            imgui.table_setup_column("󰄵 Selector", imgui.TABLE_COLUMN_NO_HIDE | imgui.TABLE_COLUMN_NO_SORT | imgui.TABLE_COLUMN_NO_RESIZE | imgui.TABLE_COLUMN_NO_REORDER, init_width_or_weight=checkbox_width)  # 0
            imgui.table_setup_column("󰖠 Eye Tracker", imgui.TABLE_COLUMN_NO_RESIZE)  # 1
            imgui.table_setup_column("󱀩 Status", imgui.TABLE_COLUMN_NO_RESIZE | (imgui.TABLE_COLUMN_DEFAULT_HIDE if self.in_adder_popup else 0))  # 2
            imgui.table_setup_column("󰌖 Name", imgui.TABLE_COLUMN_WIDTH_STRETCH | imgui.TABLE_COLUMN_DEFAULT_SORT | imgui.TABLE_COLUMN_NO_HIDE | imgui.TABLE_COLUMN_NO_RESIZE, 100.)  # 3
            imgui.table_setup_column("󰀓 Participant", imgui.TABLE_COLUMN_NO_RESIZE)  # 4
            imgui.table_setup_column("󰨸 Project", imgui.TABLE_COLUMN_DEFAULT_HIDE | imgui.TABLE_COLUMN_NO_RESIZE)  # 5
            imgui.table_setup_column("󰁫 Duration", imgui.TABLE_COLUMN_NO_RESIZE)  # 6
            imgui.table_setup_column("󱛡 Recording Start", imgui.TABLE_COLUMN_DEFAULT_HIDE | imgui.TABLE_COLUMN_NO_RESIZE)  # 7
            imgui.table_setup_column("󰷎 Working Directory", imgui.TABLE_COLUMN_DEFAULT_HIDE | imgui.TABLE_COLUMN_NO_RESIZE)  # 8
            imgui.table_setup_column("󰮟 Source Directory", imgui.TABLE_COLUMN_DEFAULT_HIDE | imgui.TABLE_COLUMN_NO_RESIZE)  # 9
            imgui.table_setup_column("󰆙 Firmware Version", imgui.TABLE_COLUMN_DEFAULT_HIDE | imgui.TABLE_COLUMN_NO_RESIZE)  # 10
            imgui.table_setup_column("󰁱 Glasses Serial", imgui.TABLE_COLUMN_DEFAULT_HIDE | imgui.TABLE_COLUMN_NO_RESIZE)  # 11
            imgui.table_setup_column("󰁱 Recording Unit Serial", imgui.TABLE_COLUMN_DEFAULT_HIDE | imgui.TABLE_COLUMN_NO_RESIZE)  # 12
            imgui.table_setup_column("󰆙 Recording Software Version", imgui.TABLE_COLUMN_DEFAULT_HIDE | imgui.TABLE_COLUMN_NO_RESIZE)  # 13
            imgui.table_setup_column("󰁱 Scene Camera Serial", imgui.TABLE_COLUMN_DEFAULT_HIDE | imgui.TABLE_COLUMN_NO_RESIZE)  # 14

            # Enabled columns
            if imgui.table_get_column_flags(0) & imgui.TABLE_COLUMN_IS_ENABLED:
                imgui.table_setup_scroll_freeze(1, 1)  # Sticky column headers and selector row
            else:
                imgui.table_setup_scroll_freeze(0, 1)  # Sticky column headers

            # Sorting
            sort_specs = imgui.table_get_sort_specs()
            sorted_recordings_ids_len = len(self.sorted_recordings_ids)
            self.sort_and_filter_recordings(sort_specs)
            if len(self.sorted_recordings_ids) < sorted_recordings_ids_len:
                # we've just filtered out some recordings from view. Deselect those
                # NB: will also be triggered when removing an item, doesn't matter
                for id in self.recordings:
                    if id not in self.sorted_recordings_ids:
                        self.selected_recordings[id] = False

            # Headers
            imgui.table_next_row(imgui.TABLE_ROW_HEADERS)
            for i in range(self._recording_list_column_count):
                imgui.table_set_column_index(i)
                column_name = imgui.table_get_column_name(i)
                if i==0:  # checkbox column: reflects whether all, some or none of visible recordings are selected, and allows selecting all or none
                    # get state
                    num_selected = sum([self.selected_recordings[id] for id in self.sorted_recordings_ids])
                    if num_selected==len(self.sorted_recordings_ids):
                        # all selected
                        multi_selected_state = 1
                    elif num_selected>0:
                        # some selected
                        multi_selected_state = 0
                    else:
                        # none selected
                        multi_selected_state = -1

                    if multi_selected_state==0:
                        imgui.internal.push_item_flag(imgui.internal.ITEM_MIXED_VALUE,True)
                    clicked, new_state = imgui.checkbox(f"###test_selected{extra}", multi_selected_state==1, frame_size=(0,0), do_vertical_align=False)
                    if multi_selected_state==0:
                        imgui.internal.pop_item_flag()

                    if clicked:
                        utils.set_all(self.selected_recordings, new_state, subset = self.sorted_recordings_ids)
                elif i in (2,):  # Hide name for small columns, just use icon
                    imgui.table_header(column_name[:1] + "###" + column_name[2:])
                else:
                    imgui.table_header(column_name[2:])

            # Loop rows
            a=.4
            style_selected_row = (*tuple(a*x+(1-a)*y for x,y in zip(globals.settings.style_accent[:3],globals.settings.style_bg[:3])), 1.)
            a=.2
            style_hovered_row  = (*tuple(a*x+(1-a)*y for x,y in zip(globals.settings.style_accent[:3],globals.settings.style_bg[:3])), 1.)
            any_selectable_clicked = False
            for idx,id in enumerate(self.sorted_recordings_ids):
                imgui.table_next_row()
                
                recording = self.recordings[id]
                num_columns_drawn = 0
                selectable_clicked = False
                checkbox_clicked, checkbox_hovered = False, False
                should_remove = False
                remove_button_hovered = False
                has_drawn_hitbox = False
                for ri in range(self._recording_list_column_count+1):
                    if not (imgui.table_get_column_flags(ri) & imgui.TABLE_COLUMN_IS_ENABLED):
                        continue
                    imgui.table_set_column_index(ri)
                    num_columns_drawn+=1

                    # Row hitbox
                    if not has_drawn_hitbox:
                        # hitbox needs to be drawn before anything else on the row so that, together with imgui.set_item_allow_overlap(), hovering button
                        # or checkbox on the row will still be correctly lead detected.
                        # this is super finicky, but works. The below together with using a height of frame_height+cell_padding_y
                        # makes the table row only cell_padding_y/2 longer. The whole row is highlighted correctly
                        cell_padding_y = imgui.style.cell_padding.y
                        cur_pos_y = imgui.get_cursor_pos_y()
                        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() - cell_padding_y/2)
                        imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 0.)
                        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0.,0.))
                        imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (0.,cell_padding_y))
                        # make selectable completely transparent
                        imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0., 0., 0., 0.)
                        imgui.push_style_color(imgui.COLOR_HEADER       , 0., 0., 0., 0.)
                        imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0., 0., 0., 0.)
                        selectable_clicked, selectable_out = imgui.selectable(f"###{id}_hitbox{extra}", self.selected_recordings[id], flags=imgui.SELECTABLE_SPAN_ALL_COLUMNS|imgui.internal.SELECTABLE_SELECT_ON_CLICK, height=frame_height+cell_padding_y)
                        # instead override table row background color
                        if selectable_out:
                            imgui.table_set_background_color(imgui.TABLE_BACKGROUND_TARGET_ROW_BG0, imgui.color_convert_float4_to_u32(*style_selected_row))
                        elif imgui.is_item_hovered():
                            imgui.table_set_background_color(imgui.TABLE_BACKGROUND_TARGET_ROW_BG0, imgui.color_convert_float4_to_u32(*style_hovered_row))
                        imgui.set_cursor_pos_y(cur_pos_y)   # instead of imgui.same_line(), we just need this part of its effect
                        imgui.set_item_allow_overlap()
                        imgui.pop_style_color(3)
                        imgui.pop_style_var(3)
                        selectable_right_clicked, should_remove_from_context_menu = \
                            self.handle_recording_hitbox_events(recording, id)
                        should_remove = should_remove_from_context_menu
                        selectable_clicked = selectable_clicked or selectable_right_clicked
                        has_drawn_hitbox = True
                        
                    if num_columns_drawn==2:
                        # (Invisible) button because it aligns the following draw calls to center vertically
                        imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 0.)
                        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0.,imgui.style.frame_padding.y))
                        imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (0.,imgui.style.item_spacing.y))
                        imgui.push_style_color(imgui.COLOR_BUTTON, 0.,0.,0.,0.)
                        imgui.button(f"###{recording.id}_id", width=imgui.FLOAT_MIN)
                        imgui.pop_style_color()
                        imgui.pop_style_var(3)
                        
                        imgui.same_line()

                    match ri:
                        case 0:
                            # Selector
                            checkbox_clicked, checkbox_out = imgui.checkbox(f"###{id}_selected{extra}", self.selected_recordings[id], frame_size=(0,0))
                            checkbox_hovered = imgui.is_item_hovered()
                        case 1:
                            # Eye Tracker
                            self.draw_eye_tracker_widget(recording, align=True)
                        case 2:
                            # Status
                            self.draw_recording_status_widget(recording)
                        case 3:
                            # Name
                            if globals.settings.show_remove_btn:
                                should_remove = self.draw_recording_remove_button(recording, id, label="󰩺") or should_remove
                                remove_button_hovered = imgui.is_item_hovered()
                                imgui.same_line()
                            self.draw_recording_name_text(recording)
                        case 4:
                            # Participant
                            imgui.text(recording.participant or "Unknown")
                        case 5:
                            # Project
                            imgui.text(recording.project or "Unknown")
                        case 6:
                            # Duration
                            imgui.text("Unknown" if (d:=recording.duration) is None else str(datetime.timedelta(seconds=d//1000)))
                        case 7:
                            # Recording Start
                            imgui.text(recording.start_time.display or "Unknown")
                        case 8:
                            # Working Directory
                            imgui.text(recording.proc_directory_name or "Unknown")
                            if imgui.is_item_hovered():
                                if recording.proc_directory_name:
                                    text = str(globals.project_path / recording.proc_directory_name)
                                else:
                                    text = 'Working dirctory not created yet'
                                draw_tooltip(text)
                        case 9:
                            # Source Directory
                            imgui.text(recording.source_directory.stem or "Unknown")
                            if recording.source_directory and imgui.is_item_hovered():
                                draw_tooltip(str(recording.source_directory))
                        case 10:
                            # Firmware Version
                            imgui.text(recording.firmware_version or "Unknown")
                        case 11:
                            # Glasses Serial
                            imgui.text(recording.glasses_serial or "Unknown")
                        case 12:
                            # Recording Unit Serial
                            imgui.text(recording.recording_unit_serial or "Unknown")
                        case 13:
                            # Recording Software Version
                            imgui.text(recording.recording_software_version or "Unknown")
                        case 14:
                            # Scene Camera Serial
                            imgui.text(recording.scene_camera_serial or "Unknown")
                    
                # handle selection logic
                any_selectable_clicked = any_selectable_clicked or selectable_clicked
                if checkbox_clicked:
                    self.selected_recordings[id] = checkbox_out
                elif selectable_clicked and not (checkbox_hovered or remove_button_hovered): # don't enter this branch if interaction is with checkbox or button on the table row
                    if imgui.get_io().key_ctrl:
                        self.selected_recordings[id] = selectable_out
                    elif imgui.get_io().key_shift:
                        if num_selected in [0,1]:
                            # select range between already selected and just clicked item
                            if num_selected==0:
                                id_other = self.sorted_recordings_ids[0]
                            else:
                                id_other = [id for id in self.selected_recordings if self.selected_recordings[id]][0]
                            idx_other = self.sorted_recordings_ids.index(id_other)
                            if idx<idx_other:
                                i1,i2 = idx,idx_other
                            else:
                                i1,i2 = idx_other,idx
                            for rid in range(i1,i2+1):
                                self.selected_recordings[self.sorted_recordings_ids[rid]] = True
                    else:
                        # deselect all other ones except clicked
                        was_selected = self.selected_recordings[id]
                        if not selectable_right_clicked or not was_selected:
                            utils.set_all(self.selected_recordings, False)
                            self.selected_recordings[id] = True if (not was_selected or num_selected>1) else selectable_out

                # handle remove action
                if should_remove:
                    # determine which to remove. If right-clicked on a selection, operate on all selected
                    if should_remove_from_context_menu:
                        ids = [rid for rid in self.selected_recordings if self.selected_recordings[rid]]
                    else:
                        ids = [id]
                    for rid in ids:
                        if self.in_adder_popup:
                            del self.recordings[rid]
                            del self.selected_recordings[rid]
                        else:
                            callbacks.remove_recording(self.recordings[rid])
                    
                    self.require_sort = True

            imgui.end_table()

            # handle click in table area outside header (not signalled by is_item_clicked())
            # and outside all selectables: deselect all
            if imgui.is_item_clicked() and not any_selectable_clicked:
                utils.set_all(self.selected_recordings, False)

    def handle_recording_hitbox_events(self, recording: Recording, id: int):
        extra = "_adder" if self.in_adder_popup else ""
        remove = False
        right_clicked = False
        # Right click = context menu
        if imgui.begin_popup_context_item(f"###{id}_context{extra}"):
            right_clicked = True
            remove = self.draw_recording_context_menu(recording, id)
            imgui.end_popup()
        return right_clicked, remove

    def draw_eye_tracker_widget(self, recording: Recording, align=False, *args, **kwargs):
        col = recording.eye_tracker.color
        imgui.push_style_color(imgui.COLOR_BUTTON, *col)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *col)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *col)
        imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 0)
        x_padding = 4
        backup_y_padding = imgui.style.frame_padding.y
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (x_padding, 0))
        if self._eye_tracker_label_width is None:
            self._eye_tracker_label_width = 0
            for eye_tracker in list(EyeTracker):
                self._eye_tracker_label_width = max(self._eye_tracker_label_width, imgui.calc_text_size(eye_tracker.name).x)
            self._eye_tracker_label_width += 2 * x_padding
        if align:
            imgui.begin_group()
            imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + backup_y_padding)
        imgui.button(f"{recording.eye_tracker.value}###{recording.id}_type", *args, width=self._eye_tracker_label_width, **kwargs)
        if align:
            imgui.end_group()
        imgui.pop_style_color(3)
        imgui.pop_style_var(2)

    def draw_recording_name_text(self, recording: Recording, *args, **kwargs):
        if globals.settings.style_color_recording_name:
            imgui.text_colored(recording.name, *globals.settings.style_accent, *args, **kwargs)
        else:
            imgui.text(recording.name, *args, **kwargs)

    def draw_recording_status_widget(self, recording: Recording, *args, **kwargs):
        if recording.status is Status.Unknown:
            imgui.text_colored("󰀨", 0.50, 0.50, 0.50, *args, **kwargs)
        elif recording.status is Status.Completed:
            imgui.text_colored("󰄳", 0.00, 0.85, 0.00, *args, **kwargs)
        elif recording.status is Status.OnHold:
            imgui.text_colored("󰏥", 0.00, 0.50, 0.95, *args, **kwargs)
        elif recording.status is Status.Abandoned:
            imgui.text_colored("󰅙", 0.87, 0.20, 0.20, *args, **kwargs)
        else:
            imgui.text("", *args, **kwargs)
        draw_hover_text(recording.status.value, text='')

    def draw_recording_remove_button(self, recording: Recording, id: int, label="", selectable=False, *args, **kwargs):
        extra = "_adder" if self.in_adder_popup else ""
        id = f"{label}###{id}_remove{extra}"
        if selectable:
            clicked = imgui.selectable(id, False, *args, **kwargs)[0]
        else:
            clicked = imgui.button(id, *args, **kwargs)
        return clicked

    def draw_recording_open_folder_button(self, recording: Recording, label="", selectable=False, source_dir=False, *args, **kwargs):
        if source_dir:
            extra = "src_"
            disable = False
            path = recording.source_directory
        else:
            extra = ""
            disable = not recording.proc_directory_name
            path = globals.project_path / recording.proc_directory_name

        id = f"{label}###{recording.id}_open_{extra}folder"
        if disable:
            utils.push_disabled()
        if selectable:
            clicked = imgui.selectable(id, False, *args, **kwargs)[0]
        else:
            clicked = imgui.button(id, *args, **kwargs)
        if disable:
            utils.pop_disabled()
        if clicked:
            callbacks.open_folder(path)
        return clicked

    def draw_recording_context_menu(self, recording: Recording, id: int):
        if not self.in_adder_popup:
            self.draw_recording_go_button(recording, label="󱄋 Process", selectable=True)
            self.draw_recording_open_folder_button(recording, label="󰷏 Open Folder", selectable=True)
            self.draw_recording_open_folder_button(recording, label="󰷏 Open Source Folder", selectable=True, source_dir=True)
        else:
            # in this context, the source folder is just the folder
            self.draw_recording_open_folder_button(recording, label="󰷏 Open Folder", selectable=True, source_dir=True)
        imgui.separator()
        return self.draw_recording_remove_button(recording, id, label="󰩺 Remove", selectable=True)

    def draw_recording_go_button(self, recording: Recording, label="", selectable=False, *args, **kwargs):
        id = f"{label}###{recording.id}_go_button"
        if selectable:
            clicked = imgui.selectable(id, False, *args, **kwargs)[0]
        else:
            clicked = imgui.button(id, *args, **kwargs)
        if clicked:
            pass # do some processing, callbacks.something(recording)
        return clicked

    def sort_and_filter_recordings(self, sort_specs_in: imgui.core._ImGuiTableSortSpecs):
        if sort_specs_in.specs_count > 0:
            sort_specs = []
            for sort_spec in sort_specs_in.specs:
                sort_specs.insert(0, SortSpec(index=sort_spec.column_index, reverse=bool(sort_spec.sort_direction - 1)))
        if sort_specs_in.specs_dirty or self.require_sort:
            ids = list(self.recordings)
            for sort_spec in sort_specs:
                match sort_spec.index:
                    case 1:     # Eye tracker
                        key = lambda id: self.recordings[id].eye_tracker.value
                    case 2:     # Status
                        key = lambda id: self.recordings[id].status.value
                    case 4:     # Participant
                        key = lambda id: self.recordings[id].participant.lower()
                    case 5:     # Project
                        key = lambda id: self.recordings[id].project.lower()
                    case 6:     # Duration
                        key = lambda id: 0 if (d:=self.recordings[id].duration) is None else d
                    case 7:     # Recording Start
                        key = lambda id: self.recordings[id].start_time.value
                    case 8:     # Directory
                        key = lambda id: self.recordings[id].proc_directory_name.lower()
                    case 9:     # Source Directory
                        key = lambda id: str(self.recordings[id].source_directory).lower()
                    case 10:    # Firmware Version
                        key = lambda id: self.recordings[id].firmware_version.lower()
                    case 11:    # Glasses Serial
                        key = lambda id: self.recordings[id].glasses_serial.lower()
                    case 12:    # Recording Unit Serial
                        key = lambda id: self.recordings[id].recording_unit_serial.lower()
                    case 13:    # Recording Software Version
                        key = lambda id: self.recordings[id].recording_software_version.lower()
                    case 14:    # Scene Camera Serial
                        key = lambda id: self.recordings[id].scene_camera_serial.lower()
                    case _:     # Name and all others
                        key = lambda id: self.recordings[id].name.lower()
                ids.sort(key=key, reverse=sort_spec.reverse)
            self.sorted_recordings_ids = ids
            for flt in self.filters:
                match flt.mode.value:
                    case FilterMode.Eye_Tracker.value:
                        key = lambda id: flt.invert != (self.recordings[id].eye_tracker is flt.match)
                    case FilterMode.Status.value:
                        key = lambda id: flt.invert != (self.recordings[id].status is flt.match)
                    case _:
                        key = None
                if key is not None:
                    self.sorted_recordings_ids = list(filter(key, self.sorted_recordings_ids))
            if self.filter_box_text:
                search = self.filter_box_text.lower()
                def key(id):
                    recording = self.recordings[id]
                    return \
                        search in recording.eye_tracker.value.lower() or \
                        search in recording.name.lower() or \
                        search in recording.participant.lower() or \
                        search in recording.project.lower()
                self.sorted_recordings_ids = list(filter(key, self.sorted_recordings_ids))
            sort_specs_in.specs_dirty = False
            self.require_sort = False


class MainGUI():
    def __init__(self):
        # Constants
        self.sidebar_size = 230
        self.window_flags: int = (
            imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_COLLAPSE |
            imgui.WINDOW_NO_TITLE_BAR |
            imgui.WINDOW_NO_SCROLLBAR |
            imgui.WINDOW_NO_SCROLL_WITH_MOUSE
        )
        self.popup_flags: int = (
            imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_COLLAPSE |
            imgui.WINDOW_NO_SAVED_SETTINGS |
            imgui.WINDOW_ALWAYS_AUTO_RESIZE
        )
        self.watermark_text         = "About"
        self.watermark_popup_text   = f"glassesValidator {globals.version}\nClick for more information"


        # Variables
        self.focused = True
        self.size_mult = 0.0
        self.last_size_mult = 1.0
        self.prev_cursor = -1
        self.recording_list: RecordingTable = None
        self.screen_pos = (0, 0)
        self.screen_size = (0, 0)
        self.new_screen_size = (0, 0)
        self.monitor = 0
        self.repeat_chars = False
        self.prev_any_hovered = None
        self.refresh_ratio_smooth = 0.0
        self.project_to_load: pathlib.Path|str = None
        self.input_chars: list[int] = []

        # Show errors in threads
        def asyncexcepthook(future: asyncio.Future):
            try:
                exc = future.exception()
            except concurrent.futures.CancelledError:
                return
            if not exc or type(exc) is msgbox.Exc:
                return
            tb = utils.get_traceback(type(exc), exc, exc.__traceback__)
            if isinstance(exc, asyncio.TimeoutError):
                utils.push_popup(msgbox.msgbox, "Processing error", f"A background process has failed:\n{type(exc).__name__}: {str(exc) or 'No further details'}\n\nPossible causes include:\n - You are running with too many workers, try lowering them in settings", MsgBox.warn, more=tb)
                return
            utils.push_popup(msgbox.msgbox, "Oops!", f"Something went wrong in an asynchronous task of a separate thread:\n\n{tb}", MsgBox.error)
        async_thread.done_callback = asyncexcepthook

        db.setup()
        self.init_imgui_glfw()

    def init_imgui_glfw(self, re_init = False):
        # tear down current context
        if re_init and (ctx := imgui.get_current_context()) is not None:
            imgui.io.ini_file_name = None   # don't store settings to ini, we already did that manually and with augmentation
            imgui.destroy_context(ctx)

        # Setup ImGui
        size, pos, is_default = self.setup_imgui()

        # Setup GLFW window
        self.setup_glfw_window(size, pos)

        # Determine what monitor we're (mostly) on, for scaling
        mon, self.monitor = utils.get_current_monitor(*self.screen_pos, *self.screen_size)
        # apply scaling
        xscale, yscale = glfw.get_monitor_content_scale(mon)
        self.size_mult = max(xscale, yscale)
        if re_init and is_default:
            glfw.set_window_size(self.window, int(self.screen_size[0]*self.size_mult), int(self.screen_size[1]*self.size_mult))
        elif self.size_mult!=self.last_size_mult:
            glfw.set_window_size(self.window, int(self.screen_size[0]/self.last_size_mult*self.size_mult), int(self.screen_size[1]/self.last_size_mult*self.size_mult))

        # make sure that styles are correctly scaled this first time we set them up
        self.last_size_mult = 1.0

        # do further setup
        self.setup_imgui_impl()
        self.setup_imgui_style()

        if not re_init:
            # this should be done only once
            self.style_imgui_functions()

    def setup_imgui(self):
        imgui.create_context()
        imgui.io = imgui.get_io()
        imgui.io.ini_file_name = str(utils.get_data_path() / "imgui.ini")
        imgui.io.config_drag_click_to_input_text = True
        size = tuple()
        pos = tuple()
        is_default = False
        try:
            # Get window size
            with open(imgui.io.ini_file_name, "r") as f:
                ini = f.read()
            imgui.load_ini_settings_from_memory(ini)
            # subpart of ini file is valid input to config parser, parse that part it
            start = ini.find("[Window][glassesValidator]")
            assert start != -1
            end = ini.find("\n\n", start)
            assert end != -1
            config = configparser.RawConfigParser()
            config.optionxform=str
            config.read_string(ini[start:end])
            try:
                size = tuple(int(x) for x in config["Window][glassesValidator"]["ScreenSize"].split(","))
            except Exception:
                pass
            try:
                pos = tuple(int(x) for x in config["Window][glassesValidator"]["ScreenPos"].split(","))
            except Exception:
                pass
            try:
                self.last_size_mult = config.getfloat("Window][glassesValidator","Scale",fallback=1.)
            except Exception:
                pass
        except Exception:
            pass
        if not len(size) == 2 or not all([isinstance(x, int) for x in size]):
            size = (1280, 720)
            is_default = True

        return size, pos, is_default

    def setup_glfw_window(self, size, pos):
        self.window: glfw._GLFWwindow = utils.impl_glfw_init(*size, "glassesValidator")
        if all([isinstance(x, int) for x in pos]) and len(pos) == 2 and utils.validate_geometry(*pos, *size):
            glfw.set_window_pos(self.window, *pos)
        self.screen_pos = glfw.get_window_pos(self.window)
        self.screen_size= glfw.get_window_size(self.window)

        icon_path = files('glassesValidator.resources.icons') / 'icon.png'
        with as_file(icon_path) as icon_file:
            self.icon_texture = imagehelper.ImageHelper(icon_file)
            self.icon_texture.reload()
            glfw.set_window_icon(self.window, 1, Image.open(icon_file))

    def setup_imgui_impl(self):
        self.impl = GlfwRenderer(self.window)
        glfw.set_char_callback(self.window, self.char_callback)
        glfw.set_window_close_callback(self.window, self.close_callback)
        glfw.set_window_focus_callback(self.window, self.focus_callback)
        glfw.set_window_pos_callback(self.window, self.pos_callback)
        glfw.set_drop_callback(self.window, self.drop_callback)
        glfw.swap_interval(globals.settings.vsync_ratio)

    def setup_imgui_style(self):
        self.refresh_fonts()

        # Load style configuration
        imgui.style = imgui.get_style()
        imgui.style.item_spacing = (imgui.style.item_spacing.y, imgui.style.item_spacing.y)
        imgui.style.frame_border_size = 1.6
        imgui.style.scrollbar_size = 10
        imgui.style.colors[imgui.COLOR_MODAL_WINDOW_DIM_BACKGROUND] = (0, 0, 0, 0.5)
        imgui.style.colors[imgui.COLOR_TABLE_BORDER_STRONG] = (0, 0, 0, 0)
        self.refresh_styles()

    def style_imgui_functions(self):
        # Custom checkbox style
        imgui._checkbox = imgui.checkbox
        def checkbox(label: str, state: bool, frame_size=None, do_vertical_align=True):
            if state:
                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND_HOVERED, *imgui.style.colors[imgui.COLOR_BUTTON_HOVERED])
                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *imgui.style.colors[imgui.COLOR_BUTTON_HOVERED])
                imgui.push_style_color(imgui.COLOR_CHECK_MARK, *imgui.style.colors[imgui.COLOR_TEXT])
            if frame_size is not None:
                frame_padding = imgui.style.frame_padding
                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, frame_size)
                imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (0.,0.))
                imgui.begin_group()
                if do_vertical_align:
                    imgui.dummy(0,frame_padding.y)
                imgui.dummy(frame_padding.x,0)
                imgui.same_line()
            result = imgui._checkbox(label, state)
            if frame_size is not None:
                imgui.end_group()
                imgui.pop_style_var(2)
            if state:
                imgui.pop_style_color(3)
            return result
        imgui.checkbox = checkbox
        # Custom combo style
        imgui._combo = imgui.combo
        def combo(*args, **kwargs):
            imgui.push_style_color(imgui.COLOR_BUTTON, *imgui.style.colors[imgui.COLOR_BUTTON_HOVERED])
            result = imgui._combo(*args, **kwargs)
            imgui.pop_style_color()
            return result
        imgui.combo = combo

    def refresh_styles(self):
        globals.settings.style_accent = \
            imgui.style.colors[imgui.COLOR_CHECK_MARK] = \
            imgui.style.colors[imgui.COLOR_TAB_ACTIVE] = \
            imgui.style.colors[imgui.COLOR_SLIDER_GRAB] = \
            imgui.style.colors[imgui.COLOR_TAB_HOVERED] = \
            imgui.style.colors[imgui.COLOR_BUTTON_ACTIVE] = \
            imgui.style.colors[imgui.COLOR_HEADER_ACTIVE] = \
            imgui.style.colors[imgui.COLOR_NAV_HIGHLIGHT] = \
            imgui.style.colors[imgui.COLOR_PLOT_HISTOGRAM] = \
            imgui.style.colors[imgui.COLOR_BUTTON_HOVERED] = \
            imgui.style.colors[imgui.COLOR_HEADER_HOVERED] = \
            imgui.style.colors[imgui.COLOR_SEPARATOR_ACTIVE] = \
            imgui.style.colors[imgui.COLOR_SEPARATOR_HOVERED] = \
            imgui.style.colors[imgui.COLOR_RESIZE_GRIP_ACTIVE] = \
            imgui.style.colors[imgui.COLOR_RESIZE_GRIP_HOVERED] = \
            imgui.style.colors[imgui.COLOR_TAB_UNFOCUSED_ACTIVE] = \
            imgui.style.colors[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE] = \
            imgui.style.colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = \
            imgui.style.colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = \
            imgui.style.colors[imgui.COLOR_TEXT_SELECTED_BACKGROUND] = \
        globals.settings.style_accent
        style_accent_dim = \
            imgui.style.colors[imgui.COLOR_TAB] = \
            imgui.style.colors[imgui.COLOR_RESIZE_GRIP] = \
            imgui.style.colors[imgui.COLOR_TAB_UNFOCUSED] = \
            imgui.style.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = \
        (*globals.settings.style_accent[:3], 0.25)
        globals.settings.style_alt_bg = \
            imgui.style.colors[imgui.COLOR_TABLE_HEADER_BACKGROUND] = \
            imgui.style.colors[imgui.COLOR_TABLE_ROW_BACKGROUND_ALT] = \
        globals.settings.style_alt_bg
        globals.settings.style_bg = \
            imgui.style.colors[imgui.COLOR_BUTTON] = \
            imgui.style.colors[imgui.COLOR_HEADER] = \
            imgui.style.colors[imgui.COLOR_FRAME_BACKGROUND] = \
            imgui.style.colors[imgui.COLOR_CHILD_BACKGROUND] = \
            imgui.style.colors[imgui.COLOR_POPUP_BACKGROUND] = \
            imgui.style.colors[imgui.COLOR_TITLE_BACKGROUND] = \
            imgui.style.colors[imgui.COLOR_WINDOW_BACKGROUND] = \
            imgui.style.colors[imgui.COLOR_SLIDER_GRAB_ACTIVE] = \
            imgui.style.colors[imgui.COLOR_SCROLLBAR_BACKGROUND] = \
        globals.settings.style_bg
        globals.settings.style_border = \
            imgui.style.colors[imgui.COLOR_BORDER] = \
            imgui.style.colors[imgui.COLOR_SEPARATOR] = \
        globals.settings.style_border
        style_corner_radius = \
            imgui.style.tab_rounding  = \
            imgui.style.grab_rounding = \
            imgui.style.frame_rounding = \
            imgui.style.child_rounding = \
            imgui.style.popup_rounding = \
            imgui.style.window_rounding = \
            imgui.style.scrollbar_rounding = \
        globals.settings.style_corner_radius * self.last_size_mult
        globals.settings.style_text = \
            imgui.style.colors[imgui.COLOR_TEXT] = \
        globals.settings.style_text
        globals.settings.style_text_dim = \
            imgui.style.colors[imgui.COLOR_TEXT_DISABLED] = \
        globals.settings.style_text_dim

        fac = self.size_mult/self.last_size_mult
        if hasattr(imgui,'scale_all_sizes'):
            imgui.scale_all_sizes(fac)
        else:
            # basically a manual implementation of scale_all_sizes()
            # although it actually does something extra in that it also
            # scales border_sizes, which scale_all_sizes() does not seem
            # to do
            imgui.style.window_padding = imgui.Vec2(*[x*fac for x in imgui.style.window_padding])
            imgui.style.window_rounding = imgui.style.window_rounding*fac
            imgui.style.window_border_size = imgui.style.window_border_size*fac
            imgui.style.window_min_size = imgui.Vec2(*[x*fac for x in imgui.style.window_min_size])
            imgui.style.child_rounding = imgui.style.child_rounding*fac
            imgui.style.child_border_size = imgui.style.child_border_size*fac
            imgui.style.popup_rounding = imgui.style.popup_rounding*fac
            imgui.style.popup_border_size = imgui.style.popup_border_size*fac
            imgui.style.frame_padding = imgui.Vec2(*[x*fac for x in imgui.style.frame_padding])
            imgui.style.frame_rounding = imgui.style.frame_rounding*fac
            imgui.style.frame_border_size = imgui.style.frame_border_size*fac
            imgui.style.item_spacing = imgui.Vec2(*[x*fac for x in imgui.style.item_spacing])
            imgui.style.item_inner_spacing = imgui.Vec2(*[x*fac for x in imgui.style.item_inner_spacing])
            imgui.style.cell_padding = imgui.Vec2(*[x*fac for x in imgui.style.cell_padding])
            imgui.style.touch_extra_padding = imgui.Vec2(*[x*fac for x in imgui.style.touch_extra_padding])
            imgui.style.indent_spacing = imgui.style.indent_spacing*fac
            imgui.style.columns_min_spacing = imgui.style.columns_min_spacing*fac
            imgui.style.scrollbar_size = imgui.style.scrollbar_size*fac
            imgui.style.scrollbar_rounding = imgui.style.scrollbar_rounding*fac
            imgui.style.grab_min_size = imgui.style.grab_min_size*fac
            imgui.style.grab_rounding = imgui.style.grab_rounding*fac
            imgui.style.log_slider_deadzone = imgui.style.log_slider_deadzone*fac
            imgui.style.tab_rounding = imgui.style.tab_rounding*fac
            imgui.style.tab_border_size = imgui.style.tab_border_size*fac
            imgui.style.tab_min_width_for_close_button = imgui.style.tab_min_width_for_close_button*fac
            imgui.style.display_window_padding = imgui.Vec2(*[x*fac for x in imgui.style.display_window_padding])
            imgui.style.display_safe_area_padding = imgui.Vec2(*[x*fac for x in imgui.style.display_safe_area_padding])
            imgui.style.mouse_cursor_scale = imgui.style.mouse_cursor_scale*fac

        self.last_size_mult = self.size_mult


    def refresh_fonts(self):
        imgui.io.fonts.clear()
        max_tex_size = gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)
        imgui.io.fonts.texture_desired_width = max_tex_size
        win_w, win_h = glfw.get_window_size(self.window)
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        font_scaling_factor = max(fb_w / win_w, fb_h / win_h)
        imgui.io.font_global_scale = 1 / font_scaling_factor
        karla_font = files('glassesValidator.resources.fonts') / 'Karla-Regular.ttf'
        noto_font = files('glassesValidator.resources.fonts') / 'NotoSans-Regular.ttf'
        mdi_font = [f for f in files('glassesValidator.resources.fonts').iterdir() if pathlib.PurePath(f).match("materialdesignicons-webfont*.ttf")][0]
        karla_config = imgui.core.FontConfig(oversample_h=3, oversample_v=3)
        noto_config = imgui.core.FontConfig(merge_mode=True, oversample_h=3, oversample_v=3)
        mdi_config = imgui.core.FontConfig(merge_mode=True, glyph_offset_y=1*self.size_mult)
        karla_range = imgui.core.GlyphRanges([0x1, 0x131, 0])
        noto_range = imgui.core.GlyphRanges([0x1, 0x10663, 0])
        mdi_range = imgui.core.GlyphRanges([0xf0000, 0xf2000, 0])
        msgbox_range = imgui.core.GlyphRanges([0xf02d7, 0xf02d7, 0xf02fc, 0xf02fc, 0xf11ce, 0xf11ce, 0xf0029, 0xf0029, 0])
        size_18 = 18 * font_scaling_factor * self.size_mult
        size_28 = 28 * font_scaling_factor * self.size_mult
        size_69 = 69 * font_scaling_factor * self.size_mult
        # Default font + more glyphs + icons
        with (
            as_file(karla_font) as karla_path,
            as_file(noto_font) as noto_path,
            as_file(mdi_font) as mdi_path
        ):
            imgui.io.fonts.add_font_from_file_ttf(str(karla_path), size_18, font_config=karla_config, glyph_ranges=karla_range)
            imgui.io.fonts.add_font_from_file_ttf(str(noto_path),  size_18, font_config=noto_config,  glyph_ranges=noto_range)
            imgui.io.fonts.add_font_from_file_ttf(str(mdi_path),   size_18, font_config=mdi_config,   glyph_ranges=mdi_range)
            # Big font + more glyphs
            self.big_font = imgui.io.fonts.add_font_from_file_ttf(str(karla_path), size_28, font_config=karla_config, glyph_ranges=karla_range)
            imgui.io.fonts.add_font_from_file_ttf(                str(noto_path),  size_28, font_config=noto_config,  glyph_ranges=noto_range)
            imgui.io.fonts.add_font_from_file_ttf(                str(mdi_path),   size_28, font_config=mdi_config,   glyph_ranges=mdi_range)
            # MsgBox type icons
            self.icon_font = msgbox.icon_font = imgui.io.fonts.add_font_from_file_ttf(str(mdi_path), size_69, glyph_ranges=msgbox_range)
        try:
            tex_width, tex_height, pixels = imgui.io.fonts.get_tex_data_as_rgba32()
        except SystemError:
            tex_height = 1
            max_tex_size = 0
        if tex_height > max_tex_size:
            self.size_mult = 1.0
            return self.refresh_fonts()
        self.impl.refresh_font_texture()
        if self.recording_list is not None:
            self.recording_list.font_changed()

    def char_callback(self, window: glfw._GLFWwindow, char: int):
        self.impl.char_callback(window, char)
        self.input_chars.append(char)

    def close_callback(self, window: glfw._GLFWwindow):
        globals.should_exit = True

    def focus_callback(self, window: glfw._GLFWwindow, focused: int):
        self.focused = focused

    def pos_callback(self, window: glfw._GLFWwindow, x: int, y: int):
        if not glfw.get_window_attrib(self.window, glfw.ICONIFIED):
            self.screen_pos = (x, y)

            # check if we moved to another monitor
            mon, mon_id = utils.get_current_monitor(*self.screen_pos, *self.screen_size)
            if mon_id != self.monitor:
                self.monitor = mon_id
                # update scaling
                xscale, yscale = glfw.get_monitor_content_scale(mon)
                if scale := max(xscale, yscale):
                    self.size_mult = scale
                    # resize window if needed
                    if self.size_mult != self.last_size_mult:
                        self.new_screen_size = int(self.screen_size[0]/self.last_size_mult*self.size_mult), int(self.screen_size[1]/self.last_size_mult*self.size_mult)

    def try_load_project(self, path: str | pathlib.Path, creating = False):
        if isinstance(path,list):
            if not path:
                utils.push_popup(msgbox.msgbox, "Project opening error", "A single project directory should be provided. None provided so cannot open.", MsgBox.error, more="Dropped paths:\n"+('\n'.join([str(p) for p in paths])))
            elif len(path)>1:
                utils.push_popup(msgbox.msgbox, "Project opening error", "Only a single project directory should be provided, but {len(path)} were provided. Cannot open multiple projects.", MsgBox.error, more="Dropped paths:\n"+('\n'.join([str(p) for p in paths])))
            else:
                path = path[0]
        path = pathlib.Path(path)
        
        if utils.is_project_folder(path):
            if creating:
                buttons = {
                    "󰄬 Yes": lambda: self.load_project(path),
                    "󰜺 No": None
                }
                utils.push_popup(msgbox.msgbox, "Create new project", "The selected folder is already a project folder.\nDo you want to open it?", MsgBox.question, buttons)
            else:
                self.load_project(path)
        elif any(path.iterdir()):
            if creating:
                utils.push_popup(msgbox.msgbox, "Project creation error", "The selected folder is not empty. Cannot be used to create a project folder.", MsgBox.error)
            else:
                utils.push_popup(msgbox.msgbox, "Project opening error", "The selected folder is not a project folder. Cannot open.", MsgBox.error)
        else:
            def init_project_and_ask():
                utils.init_project_folder(path, self.save_imgui_ini)
                buttons = {
                    "󰄬 Yes": lambda: self.load_project(path),
                    "󰜺 No": None
                }
                utils.push_popup(msgbox.msgbox, "Open new project", "Do you want to open the new project folder?", MsgBox.question, buttons)
            if creating:
                init_project_and_ask()
            else:
                buttons = {
                    "󰄬 Yes": lambda: init_project_and_ask(),
                    "󰜺 No": None
                }
                utils.push_popup(msgbox.msgbox, "Create new project", "The selected folder is empty. Do you want to use it as a new project folder?", MsgBox.warn, buttons)

    def drop_callback(self, window: glfw._GLFWwindow, items: list[str]):
        paths = [pathlib.Path(item) for item in items]
        if globals.popup_stack and isinstance(picker := globals.popup_stack[-1], filepicker.FilePicker):
            path = paths[0]
            if (picker.dir_picker and path.is_dir()) or (not picker.dir_picker and path.is_file()):
                picker.selected = str(path)
                if picker.callback:
                    picker.callback(picker.selected)
                picker.active = False
        else:
            if globals.project_path is not None:
                callbacks.add_recordings(paths)
            else:
                if len(paths)!=1 or not (path := paths[0]).is_dir():
                    utils.push_popup(msgbox.msgbox, "Project opening error", "Only a single project directory should be drag-dropped on the glassesValidator GUI.", MsgBox.error, more="Dropped paths:\n"+('\n'.join([str(p) for p in paths])))
                else:
                    # load project
                    self.try_load_project(path)

    def scaled(self, size: int | float):
        return size * self.size_mult

    def load_project(self, folder: pathlib.Path):
        if globals.project_path==folder:
            utils.push_popup(msgbox.msgbox, "Project opening error", "The selected folder is the currently opened project folder. Not re-opened.", MsgBox.error)
        else:
            self.project_to_load = folder

    def unload_project(self):
        self.project_to_load = ""

    def reload_interface(self):
        globals.project_path = None if self.project_to_load=="" else self.project_to_load
        db.setup()
        self.init_imgui_glfw(re_init=True)
        if globals.project_path is not None:
            self.recording_list = RecordingTable(globals.recordings, globals.selected_recordings)
        self.project_to_load = None

    def main_loop(self):
        scroll_energy = 0.0
        have_set_window_size = False
        while not glfw.window_should_close(self.window) and self.project_to_load is None:
            # for repeating characters that were input while bottom bar didn't have input focus
            if self.repeat_chars:
                for char in self.input_chars:
                    imgui.io.add_input_character(char)
                self.repeat_chars = False
            self.input_chars.clear()

            glfw.poll_events()
            self.impl.process_inputs()
            # if there's a queued window resize, execute
            if self.new_screen_size[0]!=0 and self.new_screen_size!=self.screen_size:
                glfw.set_window_size(self.window, *self.new_screen_size)
                glfw.poll_events()
            if not self.focused and glfw.get_window_attrib(self.window, glfw.HOVERED):
                # GlfwRenderer (self.impl) resets cursor pos if not focused, making it unresponsive
                imgui.io.mouse_pos = glfw.get_cursor_pos(self.window)
            if self.focused or globals.settings.render_when_unfocused:
                # Scroll modifiers (must be before new_frame())
                imgui.io.mouse_wheel *= globals.settings.scroll_amount
                if globals.settings.scroll_smooth:
                    scroll_energy += imgui.io.mouse_wheel
                    if abs(scroll_energy) > 0.1:
                        scroll_now = scroll_energy * imgui.io.delta_time * globals.settings.scroll_smooth_speed
                        scroll_energy -= scroll_now
                    else:
                        scroll_now = 0.0
                        scroll_energy = 0.0
                    imgui.io.mouse_wheel = scroll_now

                # Reactive cursors
                cursor = imgui.get_mouse_cursor()
                any_hovered = imgui.is_any_item_hovered()
                if cursor != self.prev_cursor or any_hovered != self.prev_any_hovered:
                    shape = glfw.ARROW_CURSOR
                    if cursor == imgui.MOUSE_CURSOR_TEXT_INPUT:
                        shape = glfw.IBEAM_CURSOR
                    elif any_hovered:
                        shape = glfw.HAND_CURSOR
                    glfw.set_cursor(self.window, glfw.create_standard_cursor(shape))
                    self.prev_cursor = cursor
                    self.prev_any_hovered = any_hovered

                # check selection should be cancelled
                if imgui.io.keys_down[glfw.KEY_ESCAPE] and not globals.popup_stack:
                    for r in globals.selected_recordings:
                        globals.selected_recordings[r] = False

                imgui.new_frame()

                imgui.set_next_window_position(0, 0, imgui.ONCE)
                if (size := imgui.io.display_size) != self.screen_size or not have_set_window_size:
                    imgui.set_next_window_size(*size, imgui.ALWAYS)
                    self.screen_size = [int(x) for x in size]
                    have_set_window_size = True

                imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0)
                imgui.begin("glassesValidator", closable=False, flags=self.window_flags)
                imgui.pop_style_var()

                text = self.watermark_text
                _3 = self.scaled(3)
                _6 = self.scaled(6)
                text_size = imgui.calc_text_size(text)
                text_x = size.x - text_size.x - _6
                text_y = size.y - text_size.y - _6

                if globals.project_path is not None:
                    sidebar_size = self.scaled(self.sidebar_size)
                    
                    imgui.begin_child("###main_frame", width=-(sidebar_size+self.scaled(4)))
                    imgui.begin_child("###recording_list_frame", height=-imgui.get_frame_height_with_spacing(), flags=imgui.WINDOW_HORIZONTAL_SCROLLING_BAR)
                    self.recording_list.draw()
                    imgui.end_child()
                    imgui.begin_child("###bottombar_frame")
                    self.recording_list.filter_box_text, self.recording_list.require_sort = \
                        self.draw_bottombar(self.recording_list.filter_box_text, self.recording_list.require_sort)
                    imgui.end_child()
                    imgui.end_child()

                    imgui.same_line(spacing=self.scaled(4))
                    imgui.begin_child("###sidebar_frame", width=sidebar_size - 1, height=-text_size.y)
                    self.draw_sidebar()
                    imgui.end_child()
                else:
                    self.draw_unopened_interface()

                imgui.set_cursor_screen_pos((text_x - _3, text_y))
                if imgui.invisible_button("###watermark_btn", width=text_size.x + _6, height=text_size.y + _3):
                    utils.push_popup(self.draw_about_popup)
                imgui.set_cursor_screen_pos((text_x, text_y))
                imgui.text(text)
                draw_hover_text(self.watermark_popup_text, text='')

                open_popup_count = 0
                for popup in globals.popup_stack:
                    if hasattr(popup, "tick"):
                        popup_func = popup.tick
                    else:
                        popup_func = popup
                    opened, closed = popup_func()
                    if closed:
                        globals.popup_stack.remove(popup)
                    open_popup_count += opened
                # Popups are closed all at the end to allow stacking
                for _ in range(open_popup_count):
                    imgui.end_popup()

                imgui.end()

                imgui.render()
                self.impl.render(imgui.get_draw_data())
                if self.size_mult != self.last_size_mult:
                    self.refresh_fonts()
                    self.refresh_styles()
                glfw.swap_buffers(self.window)  # Also waits idle time
            else:
                time.sleep(1 / 3)

        # clean up
        self.save_imgui_ini()
        self.impl.shutdown()
        glfw.terminate()
        db.shutdown()

        if self.project_to_load is not None:
            self.reload_interface()
            return True     # signal to run a fresh main loop instance
        else:
            return False


    def save_imgui_ini(self, path: str | pathlib.Path = None):
        if path is None:
            path = imgui.io.ini_file_name
        imgui.save_ini_settings_to_disk(str(path))
        ini = imgui.save_ini_settings_to_memory()

        # add some of our own stuff we want to persist
        try:
            # subpart of ini file is valid input to config parser, parse that part it
            start = ini.find("[Window][glassesValidator]")
            assert start != -1
            end = ini.find("\n\n", start)
            assert end != -1
            config = configparser.RawConfigParser()
            config.optionxform=str
            config.read_string(ini[start:end])
            # set what we want to persist
            config["Window][glassesValidator"]["ScreenSize"] = f"{self.screen_size[0]},{self.screen_size[1]}"
            config["Window][glassesValidator"]["ScreenPos"] = f"{self.screen_pos[0]},{self.screen_pos[1]}"
            config["Window][glassesValidator"]["Scale"] = f"{self.size_mult}"
            # write to string
            with io.StringIO() as ss:
                config.write(ss)
                ss.seek(0) # rewind
                config = ss.read()
            # replace section of ini with our expanded version
            ini = ini[:start] + config + ini[end+2:]
            # now write to file
            with open(str(path), "w") as f:
                f.write(ini)
        except Exception:
            pass    # already saved with imgui.save_ini_settings_to_disk above

    def get_folder_picker(self, creating=False, select_for_add=False):
        def select_callback(selected):
            if selected:
                if select_for_add:
                    callbacks.add_recordings(selected)
                else:
                    self.try_load_project(selected,creating=creating)
        picker = filepicker.DirPicker("Select or drop project folder", callback=select_callback)
        picker.show_only_dirs = True
        return picker

    def draw_unopened_interface(self):
        avail      = imgui.get_content_region_available()
        but_width  = self.scaled(200)
        but_height = self.scaled(100)

        but_x = (avail.x - 2*but_width - 10*imgui.style.item_spacing.x) / 2
        but_y = (avail.y - but_height) / 2

        imgui.push_font(self.big_font)
        text = "Drag and drop a glassesValidator folder or use the below buttons"
        size = imgui.calc_text_size(text)
        imgui.set_cursor_pos_x((avail.x - size.x) / 2)
        imgui.set_cursor_pos_y(( but_y  - size.y) / 2)
        imgui.text(text)
        imgui.pop_font()

        imgui.set_cursor_pos_x(but_x)
        imgui.set_cursor_pos_y(but_y)
        
        if imgui.button("󰮝 New project", width=but_width, height=but_height):
            utils.push_popup(self.get_folder_picker(creating=True))
        imgui.same_line(spacing=10*imgui.style.item_spacing.x)
        if imgui.button("󰷏 Open project", width=but_width, height=but_height):
            utils.push_popup(self.get_folder_picker())

    def draw_select_eye_tracker_popup(self, combo_value, eye_tracker):
        spacing = 2 * imgui.style.item_spacing.x
        icon = "󰋗"
        color = (0.45, 0.09, 1.00)
        imgui.push_font(self.icon_font)
        icon_size = imgui.calc_text_size(icon)
        imgui.text_colored(icon, *color)
        imgui.pop_font()
        imgui.same_line(spacing=spacing)

        imgui.begin_group()
        imgui.dummy(0,2*imgui.style.item_spacing.y)
        imgui.text_unformatted("For which eye tracker would you like to import recordings?")
        imgui.dummy(0,3*imgui.style.item_spacing.y)
        full_width = imgui.get_content_region_available_width()
        imgui.push_item_width(full_width*.4)
        imgui.set_cursor_pos_x(full_width*.3)
        changed, combo_value = imgui.combo("###select_eye_tracker", combo_value, eye_tracker_names)
        imgui.pop_item_width()
        imgui.dummy(0,2*imgui.style.item_spacing.y)

        imgui.end_group()
        imgui.same_line(spacing=spacing)
        imgui.dummy(0, 0)

        if changed:
            eye_tracker = EyeTracker(eye_tracker_names[combo_value])

        return combo_value, eye_tracker

    def draw_preparing_recordings_for_import_popup(self, eye_tracker):
        spacing = 2 * imgui.style.item_spacing.x
        icon = "󰋗"
        color = (0.45, 0.09, 1.00)
        imgui.push_font(self.icon_font)
        icon_size = imgui.calc_text_size(icon)
        imgui.text_colored(icon, *color)
        imgui.pop_font()
        imgui.same_line(spacing=spacing)

        imgui.begin_group()
        imgui.dummy(0,2*imgui.style.item_spacing.y)
        text = f'Searching the path(s) you provided for {eye_tracker.value} recordings.'
        imgui.text_unformatted(text)
        imgui.dummy(0,3*imgui.style.item_spacing.y)
        text_size = imgui.calc_text_size(text)
        spinner_radii = [x*self.size_mult for x in [22, 16, 10]]
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x()+(text_size.x-2*spinner_radii[0])/2)
        utils.draw_spinner('waitSpinner', *spinner_radii, 3.5*self.size_mult, c1=imgui.color_convert_float4_to_u32(*globals.settings.style_text), c2=imgui.color_convert_float4_to_u32(*globals.settings.style_accent), c3=imgui.color_convert_float4_to_u32(*globals.settings.style_text))
        imgui.dummy(0,2*imgui.style.item_spacing.y)
        imgui.end_group()

        imgui.same_line(spacing=spacing)
        imgui.dummy(0, 0)

    def draw_select_recordings_to_import(self, recording_list: RecordingTable):
        spacing = 2 * imgui.style.item_spacing.x
        imgui.same_line(spacing=spacing)

        imgui.text_unformatted("Select which recordings you would like to import.")
        imgui.dummy(0,1*imgui.style.item_spacing.y)

        imgui.begin_child("###main_frame_adder", height=min(self.scaled(300),(len(recording_list.recordings)+2)*imgui.get_frame_height_with_spacing()), width=self.scaled(800))
        imgui.begin_child("###recording_list_frame_adder", height=-imgui.get_frame_height_with_spacing(), flags=imgui.WINDOW_HORIZONTAL_SCROLLING_BAR)
        recording_list.draw()
        imgui.end_child()
        imgui.begin_child("###bottombar_frame_adder")
        recording_list.filter_box_text, recording_list.require_sort = \
            self.draw_bottombar(recording_list.filter_box_text, recording_list.require_sort, in_adder_popup=True)
        imgui.end_child()
        imgui.end_child()

        
        imgui.same_line(spacing=spacing)
        imgui.dummy(0,6*imgui.style.item_spacing.y)


    def draw_about_popup(self):
        def popup_content():
            _60 = self.scaled(60)
            _230 = self.scaled(230)
            imgui.begin_group()
            imgui.dummy(_60, _230)
            imgui.same_line()
            self.icon_texture.render(_230, _230, rounding=globals.settings.style_corner_radius)
            imgui.same_line()
            imgui.begin_group()
            imgui.push_font(self.big_font)
            imgui.text("glassesValidator")
            imgui.pop_font()
            imgui.text(f"Version {globals.version}")
            imgui.text("Made by Diederick C. Niehorster")
            imgui.text("")
            imgui.text(f"󰌠 Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
            imgui.text(f"OpenGL {'.'.join(str(gl.glGetInteger(num)) for num in (gl.GL_MAJOR_VERSION, gl.GL_MINOR_VERSION))},  󰌠 {OpenGL.__version__}")
            imgui.text(f"GLFW {'.'.join(str(num) for num in glfw.get_version())},  󰌠 {glfw.__version__}")
            imgui.text(f"ImGui {imgui.get_version()},  󰌠 {imgui.__version__}")
            if globals.os is Os.Linux:
                imgui.text(f"{platform.system()} {platform.release()}")
            elif globals.os is Os.Windows:
                imgui.text(f"{platform.system()} {platform.release()} {platform.version()}")
            elif globals.os is Os.MacOS:
                imgui.text(f"{platform.system()} {platform.release()}")
            imgui.end_group()
            imgui.same_line()
            imgui.dummy(_60, _230)
            imgui.end_group()
            imgui.spacing()
            width = imgui.get_content_region_available_width()
            btn_width = (width - 2 * imgui.style.item_spacing.x) / 3
            if imgui.button("󰌠 PyPI", width=btn_width):
                glfw.set_clipboard_string(self.window, globals.pypi_page)
            draw_hover_text(text='', hover_text="Click to copy link to clipboard")
            imgui.same_line()
            if imgui.button("󰊤 GitHub repo", width=btn_width):
                glfw.set_clipboard_string(self.window, globals.github_page)
            draw_hover_text(text='', hover_text="Click to copy link to clipboard")
            imgui.same_line()
            if imgui.button("󰖟 Researcher homepage", width=btn_width):
                glfw.set_clipboard_string(self.window, globals.developer_page)
            draw_hover_text(text='', hover_text="Click to copy link to clipboard")
            
            imgui.spacing()
            imgui.spacing()
            imgui.push_text_wrap_pos(width)
            imgui.text("This software is licensed under the 3rd revision of the GNU General Public License (GPLv3) and is provided to you for free. "
                       "Furthermore, due to its license, it is also free as in freedom: you are free to use, study, modify and share this software "
                       "in whatever way you wish as long as you keep the same license.")
            imgui.spacing()
            imgui.spacing()
            imgui.text("If you find bugs or have some feedback, please do let me know on GitHub (using issues or pull requests).")
            imgui.spacing()
            imgui.spacing()
            imgui.dummy(0, self.scaled(10))
            imgui.push_font(self.big_font)
            size = imgui.calc_text_size("Reference")
            imgui.set_cursor_pos_x((width - size.x + imgui.style.scrollbar_size) / 2)
            imgui.text("Reference")
            imgui.pop_font()
            imgui.spacing()
            imgui.spacing()
            
            imgui.text(globals.reference)
            if imgui.begin_popup_context_item(f"###refresh_context"):
                # Right click = more options context menu
                if imgui.selectable("󱓷 APA", False)[0]:
                    glfw.set_clipboard_string(self.window, globals.reference)
                if imgui.selectable("󰟀 BibTeX", False)[0]:
                    glfw.set_clipboard_string(self.window, globals.reference_bibtex)
                imgui.end_popup()
            draw_hover_text(text='', hover_text="Right-click to copy citation to clipboard")

            imgui.pop_text_wrap_pos()
        return utils.popup("About glassesValidator", popup_content, closable=True, outside=True)

    def draw_bottombar(self, filter_box_text: str, require_sort: bool, in_adder_popup: bool = False):
        extra = "_adder" if in_adder_popup else ""
        imgui.set_next_item_width(-imgui.FLOAT_MIN)
        changed = False
        if (not globals.popup_stack or in_adder_popup) and not imgui.is_any_item_active() and (self.input_chars or any(imgui.io.keys_down)):
            if imgui.is_key_pressed(glfw.KEY_BACKSPACE):
                filter_box_text = filter_box_text[:-1]
                changed = True
            # some character was input while bottom bar didn't have input focus, route to bottom bar
            if self.input_chars:
                self.repeat_chars = True
            imgui.set_keyboard_focus_here()
        _, value = imgui.input_text_with_hint(f"###bottombar{extra}", "Start typing to filter the list", filter_box_text, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
        if imgui.begin_popup_context_item(f"###bottombar_context{extra}"):
            # Right click = more options context menu
            if imgui.selectable("󰆒 Paste", False)[0]:
                value += str(glfw.get_clipboard_string(self.window) or b"", encoding="utf-8")
            imgui.separator()
            if imgui.selectable("󰋽 More info", False)[0]:
                utils.push_popup(
                    msgbox.msgbox, "About the bottom bar",
                    "This is the filter bar. By typing inside it you can search your recording list inside the eye tracker, name, participant and project properties.",
                    MsgBox.info
                )
            imgui.end_popup()
        if changed or value != filter_box_text:
            filter_box_text = value
            require_sort = True

        return filter_box_text, require_sort

    def start_settings_section(self, name: str, right_width: int | float, collapsible=True):
        if collapsible:
            header = imgui.collapsing_header(name)[0]
        else:
            header = True
        opened = header and imgui.begin_table(f"###settings_{name}", column=2, flags=imgui.TABLE_NO_CLIP)
        if opened:
            imgui.table_setup_column(f"###settings_{name}_left", imgui.TABLE_COLUMN_WIDTH_STRETCH)
            imgui.table_setup_column(f"###settings_{name}_right", imgui.TABLE_COLUMN_WIDTH_FIXED)
            imgui.table_next_row()
            imgui.table_set_column_index(1)  # Right
            imgui.dummy(right_width, 1)
            imgui.push_item_width(right_width)
        return opened

    def draw_sidebar(self):
        set = globals.settings
        right_width = self.scaled(90)
        frame_height = imgui.get_frame_height()
        checkbox_offset = right_width - frame_height
        width = imgui.get_content_region_available_width()

        height = self.scaled(100)
        # Normal button
        if imgui.button("Process all!", width=width, height=height):
            pass
        if imgui.begin_popup_context_item(f"###refresh_context"):
            # Right click = more options context menu
            if imgui.selectable("󰅸 Do stuff", False)[0]:
                pass # do some processing, callbacks.something(recording)
            imgui.end_popup()

        imgui.begin_child("Settings")

        if self.start_settings_section("Filter", right_width, collapsible=False):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(f"Total recording count: {len(self.recording_list.recordings)}")
            imgui.spacing()
            if self.recording_list.filters or self.recording_list.filter_box_text:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(f"Filtered recording count: {len(self.recording_list.sorted_recordings_ids)}")
                imgui.spacing()

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Add filter:")
            imgui.table_next_column()
            changed, value = imgui.combo("###add_filter", 0, filter_mode_names)
            if changed and value > 0:
                flt = Filter(FilterMode(filter_mode_names[value]))
                match flt.mode.value:
                    case FilterMode.Eye_Tracker.value:
                        flt.match = EyeTracker(eye_tracker_names[0])
                    case FilterMode.Status.value:
                        flt.match = Status(status_names[0])
                self.recording_list.add_filter(flt)

            for flt in self.recording_list.filters:
                imgui.spacing()
                imgui.spacing()
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(f"{flt.mode.value} filter:")
                imgui.table_next_column()
                if imgui.button(f"Remove###filter_{flt.id}_remove", width=right_width):
                    self.recording_list.remove_filter(flt.id)
                    
                if flt.mode is FilterMode.Status:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text("  Status value:")
                    imgui.table_next_column()
                    changed, value = imgui.combo(f"###filter_{flt.id}_value", status_names.index(flt.match.value), status_names)
                    if changed:
                        flt.match = Status(status_names[value])
                        self.recording_list.require_sort = True

                elif flt.mode is FilterMode.Eye_Tracker:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text("  Eye Tracker:")
                    imgui.table_next_column()
                    changed, value = imgui.combo(f"###filter_{flt.id}_value", eye_tracker_names.index(flt.match.value), eye_tracker_names)
                    if changed:
                        flt.match = EyeTracker(eye_tracker_names[value])
                        self.recording_list.require_sort = True

                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text("  Invert filter:")
                imgui.table_next_column()
                imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + checkbox_offset)
                changed, value = imgui.checkbox(f"###filter_{flt.id}_invert", flt.invert)
                if changed:
                    flt.invert = value
                    self.recording_list.require_sort = True

            imgui.end_table()
            imgui.spacing()

        if self.start_settings_section("Project", right_width):
            # for full width buttons, in lieu of column-spanning API that doesn't exist
            # interrupt table
            imgui.end_table()

            btn_width = right_width*1.5
            imgui.set_cursor_pos_x((width-btn_width)/2)
            if imgui.button("󰮝 New project", width=btn_width):
                utils.push_popup(self.get_folder_picker(creating=True))
            imgui.set_cursor_pos_x((width-btn_width)/2)
            if imgui.button("󰷏 Open project", width=btn_width):
                utils.push_popup(self.get_folder_picker())
            imgui.set_cursor_pos_x((width-btn_width)/2)
            if imgui.button("󰮞 Close project", width=btn_width):
                self.unload_project()

            imgui.set_cursor_pos_x((width-btn_width)/2)
            if imgui.button("󱃩 Add recordings", width=btn_width):
                utils.push_popup(self.get_folder_picker(select_for_add=True))
            draw_hover_text(
                "Press the \"󱃩 Add recordings\" button to select a folder or folders "
                "that will be searched for importable recordings. You will then be able "
                "to select which of the found recordings you wish to import. You can "
                "also start importing recordings by drag-dropping one or multiple "
                "folders onto glassesValidator.", text='')
                
            # continue table
            self.start_settings_section("Project", right_width, collapsible = False)
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Show remove button:")
            imgui.table_next_column()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + checkbox_offset)
            changed, value = imgui.checkbox("###show_remove_btn", set.show_remove_btn)
            if changed:
                set.show_remove_btn = value
                async_thread.run(db.update_settings("show_remove_btn"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Confirm when removing:")
            imgui.table_next_column()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + checkbox_offset)
            changed, value = imgui.checkbox("###confirm_on_remove", set.confirm_on_remove)
            if changed:
                set.confirm_on_remove = value
                async_thread.run(db.update_settings("confirm_on_remove"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Workers:")
            imgui.same_line()
            draw_hover_text(
                "Each recording is processed by a worker and each worker can handle 1 "
                "recording at a time. Having more workers means more recordings are processed "
                "simultaneously, but having too many will freeze the program. In most cases 3 "
                "workers is a good compromise."
            )
            imgui.table_next_column()
            changed, value = imgui.drag_int("###refresh_workers", set.refresh_workers, change_speed=0.5, min_value=1, max_value=100)
            set.refresh_workers = min(max(value, 1), 100)
            if changed:
                async_thread.run(db.update_settings("refresh_workers"))
                
            imgui.end_table()
            imgui.spacing()

        if self.start_settings_section("Interface", right_width):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Smooth scrolling:")
            imgui.table_next_column()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + checkbox_offset)
            changed, value = imgui.checkbox("###scroll_smooth", set.scroll_smooth)
            if changed:
                set.scroll_smooth = value
                async_thread.run(db.update_settings("scroll_smooth"))

            if not set.scroll_smooth:
                utils.push_disabled()

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Smoothness:")
            imgui.same_line()
            draw_hover_text(
                "How fast or slow the smooth scrolling animation is. Default is 8."
            )
            imgui.table_next_column()
            changed, value = imgui.drag_float("###scroll_smooth_speed", set.scroll_smooth_speed, change_speed=0.25, min_value=0.1, max_value=50)
            set.scroll_smooth_speed = min(max(value, 0.1), 50)
            if changed:
                async_thread.run(db.update_settings("scroll_smooth_speed"))

            if not set.scroll_smooth:
                utils.pop_disabled()

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Scroll mult:")
            imgui.same_line()
            draw_hover_text(
                "Multiplier for how much a single scroll event should actually scroll. Default is 1."
            )
            imgui.table_next_column()
            changed, value = imgui.drag_float("###scroll_amount", set.scroll_amount, change_speed=0.05, min_value=0.1, max_value=10, format="%.2fx")
            set.scroll_amount = min(max(value, 0.1), 10)
            if changed:
                async_thread.run(db.update_settings("scroll_amount"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Vsync ratio:")
            imgui.same_line()
            draw_hover_text(
                "Vsync means that the framerate should be synced to the one your monitor uses. The ratio modifies this behavior. "
                "A ratio of 1:0 means uncapped framerate, while all other numbers indicate the ratio between screen and app FPS. "
                "For example a ratio of 1:2 means the app refreshes every 2nd monitor frame, resulting in half the framerate."
            )
            imgui.table_next_column()
            changed, value = imgui.drag_int("###vsync_ratio", set.vsync_ratio, change_speed=0.05, min_value=0, max_value=10, format="1:%d")
            set.vsync_ratio = min(max(value, 0), 10)
            if changed:
                glfw.swap_interval(set.vsync_ratio)
                async_thread.run(db.update_settings("vsync_ratio"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Render if unfocused:")
            imgui.same_line()
            draw_hover_text(
                "glassesValidator renders its interface using ImGui and OpenGL and this means it has to render the whole interface up "
                "to hundreds of times per second (look at the framerate below). This process is as optimized as possible but it "
                "will inevitably consume some CPU and GPU resources. If you absolutely need the performance you can disable this "
                "option to stop rendering when the checker window is not focused, but keep in mind that it might lead to weird "
                "interactions and behavior."
            )
            imgui.table_next_column()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + checkbox_offset)
            changed, value = imgui.checkbox("###render_when_unfocused", set.render_when_unfocused)
            if changed:
                set.render_when_unfocused = value
                async_thread.run(db.update_settings("render_when_unfocused"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(f"Current framerate: {round(imgui.io.framerate, 2)}")
            imgui.spacing()

            imgui.end_table()
            imgui.spacing()

        if self.start_settings_section("Style", right_width):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Corner radius:")
            imgui.table_next_column()
            changed, value = imgui.drag_int("###style_corner_radius", set.style_corner_radius, change_speed=0.04, min_value=0, max_value=20, format="%d px")
            set.style_corner_radius = min(max(value, 0), 20)
            if changed:
                imgui.style.window_rounding = imgui.style.frame_rounding = imgui.style.tab_rounding = \
                imgui.style.child_rounding = imgui.style.grab_rounding = imgui.style.popup_rounding = \
                imgui.style.scrollbar_rounding = globals.settings.style_corner_radius * self.size_mult
                async_thread.run(db.update_settings("style_corner_radius"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Accent:")
            imgui.table_next_column()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + checkbox_offset)
            changed, value = imgui.color_edit3("###style_accent", *set.style_accent[:3], flags=imgui.COLOR_EDIT_NO_INPUTS)
            if changed:
                set.style_accent = (*value, 1.0)
                self.refresh_styles()
                async_thread.run(db.update_settings("style_accent"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Color recording name:")
            imgui.same_line()
            draw_hover_text(
                "If selected, recording name will be drawn in the accent color instead of the text color."
            )
            imgui.table_next_column()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + checkbox_offset)
            changed, value = imgui.checkbox("###style_color_recording_name", set.style_color_recording_name)
            if changed:
                set.style_color_recording_name = value
                async_thread.run(db.update_settings("style_color_recording_name"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Background:")
            imgui.table_next_column()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + checkbox_offset)
            changed, value = imgui.color_edit3("###style_bg", *set.style_bg[:3], flags=imgui.COLOR_EDIT_NO_INPUTS)
            if changed:
                set.style_bg = (*value, 1.0)
                self.refresh_styles()
                async_thread.run(db.update_settings("style_bg"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Alt background:")
            imgui.table_next_column()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + checkbox_offset)
            changed, value = imgui.color_edit3("###style_alt_bg", *set.style_alt_bg[:3], flags=imgui.COLOR_EDIT_NO_INPUTS)
            if changed:
                set.style_alt_bg = (*value, 1.0)
                self.refresh_styles()
                async_thread.run(db.update_settings("style_alt_bg"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Border:")
            imgui.table_next_column()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + checkbox_offset)
            changed, value = imgui.color_edit3("###style_border", *set.style_border[:3], flags=imgui.COLOR_EDIT_NO_INPUTS)
            if changed:
                set.style_border = (*value, 1.0)
                self.refresh_styles()
                async_thread.run(db.update_settings("style_border"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Text:")
            imgui.table_next_column()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + checkbox_offset)
            changed, value = imgui.color_edit3("###style_text", *set.style_text[:3], flags=imgui.COLOR_EDIT_NO_INPUTS)
            if changed:
                set.style_text = (*value, 1.0)
                self.refresh_styles()
                async_thread.run(db.update_settings("style_text"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Text dim:")
            imgui.table_next_column()
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + checkbox_offset)
            changed, value = imgui.color_edit3("###style_text_dim", *set.style_text_dim[:3], flags=imgui.COLOR_EDIT_NO_INPUTS)
            if changed:
                set.style_text_dim = (*value, 1.0)
                self.refresh_styles()
                async_thread.run(db.update_settings("style_text_dim"))

            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text("Defaults:")
            imgui.table_next_column()
            style = None
            if imgui.button("Dark", width=right_width):
                style = DefaultStyleDark
            if imgui.button("Light", width=right_width):
                style = DefaultStyleLight
            if style is not None:
                set.style_corner_radius = style.corner_radius
                set.style_accent        = hex_to_rgba_0_1(style.accent)
                set.style_alt_bg        = hex_to_rgba_0_1(style.alt_bg)
                set.style_bg            = hex_to_rgba_0_1(style.bg)
                set.style_border        = hex_to_rgba_0_1(style.border)
                set.style_text          = hex_to_rgba_0_1(style.text)
                set.style_text_dim      = hex_to_rgba_0_1(style.text_dim)
                self.refresh_styles()
                async_thread.run(db.update_settings(
                    "style_corner_radius",
                    "style_accent",
                    "style_alt_bg",
                    "style_bg",
                    "style_border",
                    "style_text",
                    "style_text_dim",
                ))

            imgui.end_table()
            imgui.spacing()

        imgui.end_child()
