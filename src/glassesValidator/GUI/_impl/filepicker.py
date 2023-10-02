import pathlib
import typing
import string
from imgui_bundle import imgui
import glfw
import sys
import os
import dataclasses
import natsort
import mimetypes
import datetime
import re

from . import globals, utils
from .structs import SortSpec

@dataclasses.dataclass
class DirEntry:
    name: str
    is_dir: bool
    full_path: pathlib.Path
    ctime: float
    mtime: float
    size: int
    mime_type: str

class FilePicker:
    flags: int = (
        imgui.WindowFlags_.no_resize |
        imgui.WindowFlags_.no_collapse |
        imgui.WindowFlags_.no_saved_settings |
        imgui.WindowFlags_.always_auto_resize
    )

    def __init__(self, title="File picker", dir_picker=False, start_dir: str | pathlib.Path = None, callback: typing.Callable = None, allow_multiple = True, custom_popup_flags=0):
        self.title = title
        self.elapsed = 0.0
        self.dir_icon = "󰉋  "
        self.file_icon = "󰈔  "
        self.callback = callback

        self.items: dict[int, DirEntry] = {}
        self.selected: dict[int, bool] = {}
        self.allow_multiple = allow_multiple
        self.msg: str = None
        self.require_sort = False
        self.sorted_items: list[int] = []
        self.last_clicked_id: int = None

        self.dir: pathlib.Path = None
        self.dir_picker = dir_picker
        self.show_only_dirs = self.dir_picker   # by default, a dir picker only shows dirs
        self.predicate = None
        self.flags = custom_popup_flags or self.flags
        self.windows = sys.platform.startswith("win")
        if self.windows:
            self.drives: list[str] = []
            self.current_drive = 0

        self.goto(start_dir or os.getcwd())

    def set_dir(self, paths: pathlib.Path | list[pathlib.Path]):
        if not isinstance(paths,list):
            paths = [paths]
        paths = [pathlib.Path(p) for p in paths]

        if len(paths)==1 and paths[0].is_dir():
            self.goto(paths[0])
        else:
            self.goto(paths[0].parent)
            # update selected
            got_one = False
            for p in paths:
                for id in self.items:
                    entry = self.items[id]
                    if entry.full_path==p and (not self.predicate or self.predicate(id)):
                        self.selected[id] = True
                        got_one = True
                        break
                if not self.allow_multiple and got_one:
                    break

    def goto(self, dir: str | pathlib.Path):
        dir = pathlib.Path(dir)
        if dir.is_file():
            dir = dir.parent
        if dir.is_dir():
            self.dir = dir
        elif self.dir is None:
            self.dir = pathlib.Path(os.getcwd())
        self.dir = self.dir.absolute()
        self.selected = {}
        self.refresh()

    def refresh(self):
        selected = [self.items[id] for id in self.items if id in self.selected and self.selected[id]]
        self.items.clear()
        self.selected.clear()
        self.msg = None
        if self.dir_picker and self.show_only_dirs and not self.predicate:
            self.predicate = lambda id: self.items[id].is_dir
        try:
            items = list(self.dir.iterdir())
            if not items:
                self.msg = "This folder is empty!"
            else:
                if self.dir_picker and self.show_only_dirs:
                    items = [i for i in items if i.is_dir()]
                if items:
                    for i,item in enumerate(items):
                        stat = item.stat()
                        self.items[i] = DirEntry(item.name,item.is_dir(),item,
                                                 stat.st_ctime,stat.st_mtime,stat.st_size,
                                                 mimetypes.guess_type(item)[0])
                        self.selected[i] = False
                else:
                    self.msg = "This folder does not contain any folders!"

        except Exception:
            self.msg = "Cannot open this folder!"

        for old in selected:
            for id in self.items:
                entry = self.items[id]
                if entry.name==old.name:
                    self.selected[id] = True
                    break

        self.require_sort = True

        if self.windows:
            self.drives.clear()
            i = -1
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if pathlib.Path(drive).exists():
                    i += 1
                    self.drives.append(drive)
                    if str(self.dir).startswith(drive):
                        self.current_drive = i

    def tick(self):
        io = imgui.get_io()

        # Auto refresh
        self.elapsed += io.delta_time
        if self.elapsed > 2:
            self.elapsed = 0.0
            self.refresh()

        # Setup popup
        if not imgui.is_popup_open(self.title):
            imgui.open_popup(self.title)
        cancelled = closed = False
        opened = 1
        pos = imgui.get_window_pos()
        size = imgui.get_window_size()
        imgui.set_next_window_pos((pos.x+size.x/2, pos.y+size.y/2), pivot=(0.5,0.5), cond=imgui.Cond_.appearing)
        if imgui.begin_popup_modal(self.title, True, flags=self.flags)[0]:
            cancelled = closed = utils.close_weak_popup()
            imgui.begin_group()
            # Up button
            if imgui.button("󰁞"):
                self.goto(self.dir.parent)
            # Drive selector
            if self.windows:
                imgui.same_line()
                imgui.set_next_item_width(imgui.get_font_size() * 4)
                changed, value = imgui.combo("##drive_selector", self.current_drive, self.drives)
                if changed:
                    self.goto(self.drives[value])
            # Location bar
            imgui.same_line()
            imgui.set_next_item_width(size.x * 0.7)
            confirmed, dir = imgui.input_text("##location_bar", str(self.dir), flags=imgui.InputTextFlags_.enter_returns_true)
            if imgui.begin_popup_context_item(f"##location_context"):
                if imgui.selectable("󰆒 Paste", False)[0] and (clip := glfw.get_clipboard_string(globals.gui.window)):
                    dir = str(clip, encoding="utf-8")
                    confirmed = True
                imgui.end_popup()
            if confirmed:
                self.goto(dir)
            # Refresh button
            imgui.same_line()
            if imgui.button("󰑐"):
                self.refresh()
            imgui.end_group()

            # entry list
            num_selected = 0
            imgui.begin_child("##folder_contents", size=(imgui.get_item_rect_size().x, size.y*0.65))
            if self.msg:
                imgui.text_unformatted(self.msg)
            else:
                table_flags = (
                    imgui.TableFlags_.scroll_x |
                    imgui.TableFlags_.scroll_y |
                    imgui.TableFlags_.hideable |
                    imgui.TableFlags_.sortable |
                    imgui.TableFlags_.resizable |
                    imgui.TableFlags_.sort_multi |
                    imgui.TableFlags_.reorderable |
                    imgui.TableFlags_.row_bg |
                    imgui.TableFlags_.sizing_fixed_fit |
                    imgui.TableFlags_.no_host_extend_y |
                    imgui.TableFlags_.no_borders_in_body_until_resize
                )
                if imgui.begin_table(f"##folder_list",column=5+self.allow_multiple,flags=table_flags):
                    frame_height = imgui.get_frame_height()

                    # Setup
                    checkbox_width = frame_height
                    if self.allow_multiple:
                        imgui.table_setup_column("Selector", imgui.TableColumnFlags_.no_hide | imgui.TableColumnFlags_.no_sort | imgui.TableColumnFlags_.no_resize | imgui.TableColumnFlags_.no_reorder, init_width_or_weight=checkbox_width)  # 0
                    imgui.table_setup_column("Name", imgui.TableColumnFlags_.default_sort | imgui.TableColumnFlags_.no_hide)  # 1
                    imgui.table_setup_column("Date created", imgui.TableColumnFlags_.default_hide)  # 2
                    imgui.table_setup_column("Date modified")  # 3
                    imgui.table_setup_column("Type")  # 4
                    imgui.table_setup_column("Size")  # 5
                    imgui.table_setup_scroll_freeze(int(self.allow_multiple), 1)  # Sticky column headers and selector row

                    sort_specs = imgui.table_get_sort_specs()
                    self.sort_items(sort_specs)

                    # Headers
                    imgui.table_next_row(imgui.TableRowFlags_.headers)
                    # checkbox column: reflects whether all, some or none of visible recordings are selected, and allows selecting all or none
                    num_selected = sum([self.selected[id] for id in self.selected])
                    if self.allow_multiple:
                        imgui.table_set_column_index(0)
                        # determine state
                        if self.predicate:
                            num_items = sum([self.predicate(id) for id in self.items])
                        else:
                            num_items = len(self.items)
                        if num_selected==0:
                            # none selected
                            multi_selected_state = -1
                        elif num_selected==num_items:
                            # all selected
                            multi_selected_state = 1
                        else:
                            # some selected
                            multi_selected_state = 0

                        if multi_selected_state==0:
                            imgui.internal.push_item_flag(imgui.internal.ItemFlags_.mixed_value, True)
                        clicked, new_state = imgui.checkbox(f"##header_checkbox", multi_selected_state==1, frame_size=(0,0), do_vertical_align=False)
                        if multi_selected_state==0:
                            imgui.internal.pop_item_flag()

                        if clicked:
                            utils.set_all(self.selected, new_state, subset = self.sorted_items, predicate=self.predicate)

                    for i in range(5):
                        imgui.table_set_column_index(i+self.allow_multiple)
                        imgui.table_header(imgui.table_get_column_name(i+self.allow_multiple))


                    # Loop rows
                    a=.4
                    style_selected_row = (*tuple(a*x+(1-a)*y for x,y in zip(globals.settings.style_accent[:3],globals.settings.style_bg[:3])), 1.)
                    a=.2
                    style_hovered_row  = (*tuple(a*x+(1-a)*y for x,y in zip(globals.settings.style_accent[:3],globals.settings.style_bg[:3])), 1.)
                    any_selectable_clicked = False
                    if self.sorted_items and self.last_clicked_id not in self.sorted_items:
                        # default to topmost if last_clicked unknown, or no longer on screen due to filter
                        self.last_clicked_id = self.sorted_items[0]
                    for id in self.sorted_items:
                        imgui.table_next_row()

                        selectable_clicked = False
                        checkbox_clicked, checkbox_hovered, checkbox_out = False, False, False
                        has_drawn_hitbox = False

                        disable_item = self.predicate and not self.predicate(id)
                        if disable_item:
                            imgui.internal.push_item_flag(imgui.internal.ItemFlags_.disabled, True)
                            imgui.push_style_var(imgui.StyleVar_.alpha, imgui.get_style().alpha * 0.5)

                        for ci in range(5+self.allow_multiple):
                            if not (imgui.table_get_column_flags(ci) & imgui.TableColumnFlags_.is_enabled):
                                continue
                            imgui.table_set_column_index(ci)

                            # Row hitbox
                            if not has_drawn_hitbox:
                                # hitbox needs to be drawn before anything else on the row so that, together with imgui.set_item_allow_overlap(), hovering button
                                # or checkbox on the row will still be correctly detected.
                                # this is super finicky, but works. The below together with using a height of frame_height+cell_padding_y
                                # makes the table row only cell_padding_y/2 longer. The whole row is highlighted correctly
                                cell_padding_y = imgui.style.cell_padding.y
                                cur_pos_y = imgui.get_cursor_pos_y()
                                imgui.set_cursor_pos_y(cur_pos_y - cell_padding_y/2)
                                imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.)
                                imgui.push_style_var(imgui.StyleVar_.frame_padding    , (0.,0.))
                                imgui.push_style_var(imgui.StyleVar_.item_spacing     , (0.,cell_padding_y))
                                # make selectable completely transparent
                                imgui.push_style_color(imgui.Col_.header_active , (0., 0., 0., 0.))
                                imgui.push_style_color(imgui.Col_.header        , (0., 0., 0., 0.))
                                imgui.push_style_color(imgui.Col_.header_hovered, (0., 0., 0., 0.))
                                selectable_clicked, selectable_out = imgui.selectable(f"##{id}_hitbox", self.selected[id], flags=imgui.SelectableFlags_.span_all_columns|imgui.SelectableFlags_.allow_overlap|imgui.internal.SelectableFlagsPrivate_.select_on_click, size=(0,frame_height+cell_padding_y))
                                # instead override table row background color
                                if selectable_out:
                                    imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg0, imgui.color_convert_float4_to_u32(style_selected_row))
                                elif imgui.is_item_hovered():
                                    imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg0, imgui.color_convert_float4_to_u32(style_hovered_row))
                                imgui.set_cursor_pos_y(cur_pos_y)   # instead of imgui.same_line(), we just need this part of its effect
                                imgui.pop_style_color(3)
                                imgui.pop_style_var(3)
                                has_drawn_hitbox = True

                            if ci==int(self.allow_multiple):
                                # (Invisible) button because it aligns the following draw calls to center vertically
                                imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.)
                                imgui.push_style_var(imgui.StyleVar_.frame_padding    , (0.,imgui.style.frame_padding.y))
                                imgui.push_style_var(imgui.StyleVar_.item_spacing     , (0.,imgui.style.item_spacing.y))
                                imgui.push_style_color(imgui.Col_.button, (0.,0.,0.,0.))
                                imgui.button(f"##{id}_id", size=(imgui.FLT_MIN,0))
                                imgui.pop_style_color()
                                imgui.pop_style_var(3)

                                imgui.same_line()

                            match ci+int(not self.allow_multiple):
                                case 0:
                                    # Selector
                                    checkbox_clicked, checkbox_out = imgui.checkbox(f"##{id}_selected", self.selected[id], frame_size=(0,0))
                                    checkbox_hovered = imgui.is_item_hovered()
                                case 1:
                                    # Name
                                    prefix = self.dir_icon if self.items[id].is_dir else self.file_icon
                                    imgui.text(prefix+self.items[id].name)
                                case 2:
                                    # Date created
                                    dt = datetime.datetime.fromtimestamp(self.items[id].ctime).strftime("%Y-%m-%d %H:%M:%S")
                                    imgui.text(dt)
                                case 3:
                                    # Date modified
                                    dt = datetime.datetime.fromtimestamp(self.items[id].mtime).strftime("%Y-%m-%d %H:%M:%S")
                                    imgui.text(dt)
                                case 4:
                                    # Type
                                    if self.items[id].mime_type:
                                        imgui.text(self.items[id].mime_type)
                                case 5:
                                    # Size
                                    if not self.items[id].is_dir:
                                        unit = 1024**2
                                        orig = "%.1f KiB" % ((1024 * self.items[id].size / unit))
                                        while True:
                                            new = re.sub(r"^(-?\d+)(\d{3})", r"\g<1>,\g<2>", orig)
                                            if orig == new:
                                                break
                                            orig = new
                                        imgui.text(new)

                        if disable_item:
                            imgui.internal.pop_item_flag()
                            imgui.pop_style_var()

                        # handle selection logic
                        # NB: any_selectable_clicked is just for handling clicks not on any item
                        any_selectable_clicked = any_selectable_clicked or selectable_clicked
                        self.last_clicked_id = utils.selectable_item_logic(
                            id, self.selected, self.last_clicked_id, self.sorted_items,
                            selectable_clicked, selectable_out, allow_multiple=self.allow_multiple,
                            overlayed_hovered=checkbox_hovered, overlayed_clicked=checkbox_clicked, new_overlayed_state=checkbox_out
                            )

                        # further deal with doubleclick on item
                        if selectable_clicked and not checkbox_hovered: # don't enter this branch if interaction is with checkbox on the table row
                            if not imgui.io.key_ctrl and not imgui.io.key_shift and imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
                                if self.items[id].is_dir:
                                    self.goto(self.items[id].full_path)
                                    break
                                elif not self.dir_picker:
                                    utils.set_all(self.selected, False)
                                    self.selected[id] = True
                                    imgui.close_current_popup()
                                    closed = True

                    last_y = imgui.get_cursor_screen_pos().y
                    imgui.end_table()

                    # handle click in table area outside header+contents:
                    # deselect all, and if right click, show popup
                    # check mouse is below bottom of last drawn row so that clicking on the one pixel empty space between selectables
                    # does not cause everything to unselect or popup to open
                    if imgui.is_item_clicked() and not any_selectable_clicked and imgui.io.mouse_pos.y>last_y:  # left mouse click (NB: table header is not signalled by is_item_clicked(), so this works correctly)
                        utils.set_all(self.selected, False)

            imgui.end_child()

            # Cancel button
            if imgui.button("󰜺 Cancel"):
                imgui.close_current_popup()
                cancelled = closed = True
            # Ok button
            imgui.same_line()
            num_selected = sum([self.selected[id] for id in self.selected])
            disable_ok = not num_selected and not self.dir_picker
            if disable_ok:
                imgui.internal.push_item_flag(imgui.internal.ItemFlags_.disabled, True)
                imgui.push_style_var(imgui.StyleVar_.alpha, imgui.get_style().alpha *  0.5)
            if imgui.button("󰄬 Ok"):
                imgui.close_current_popup()
                closed = True
            if disable_ok:
                imgui.internal.pop_item_flag()
                imgui.pop_style_var()
            # Selected text
            imgui.same_line()
            if self.dir_picker and not num_selected:
                imgui.text(f"  Selected the current directory ({self.dir if self.dir==self.dir.parent else self.dir.name})")
            elif num_selected==1:
                selected = [self.items[id] for id in self.items if id in self.selected and self.selected[id]]
                imgui.text(f"  Selected {num_selected} item ({'directory' if selected[0].is_dir else 'file'} '{selected[0].name}')")
            else:
                imgui.text(f"  Selected {num_selected} items")
        else:
            opened = 0
            cancelled = closed = True
        if closed:
            if not cancelled and self.callback:
                selected = [self.items[id].full_path for id in self.items if id in self.selected and self.selected[id]]
                if self.dir_picker and not selected:
                    selected = [self.dir]
                self.callback(selected if selected else None)
        return opened, closed

    def sort_items(self, sort_specs_in: imgui.TableSortSpecs):
        if sort_specs_in.specs_dirty or self.require_sort:
            ids = list(self.items)
            sort_specs = [sort_specs_in.get_specs(i) for i in range(sort_specs_in.specs_count)]
            for sort_spec in reversed(sort_specs):
                pass
                sort_spec = SortSpec(index=sort_spec.column_index, reverse=bool(sort_spec.get_sort_direction() - 1))
                match sort_spec.index+int(not self.allow_multiple):
                    case 2:     # Date created
                        key = lambda id: self.items[id].ctime
                    case 3:     # Date modified
                        key = lambda id: self.items[id].mtime
                    case 4:     # Type
                        key = lambda id: m if (m:=self.items[id].mime_type) else ''
                    case 5:     # Size
                        key = lambda id: self.items[id].size

                    case _:     # Name and all others
                        key = natsort.os_sort_keygen(key=lambda id: self.items[id].full_path)

                ids.sort(key=key, reverse=sort_spec.reverse)

            # finally, always sort dirs first
            ids.sort(key=lambda id: self.items[id].is_dir, reverse=True)
            self.sorted_items = ids
            sort_specs_in.specs_dirty = False
            self.require_sort = False


class DirPicker(FilePicker):
    def __init__(self, title="Directory picker", start_dir: str | pathlib.Path = None, callback: typing.Callable = None, allow_multiple = True, custom_popup_flags=0):
        super().__init__(title=title, dir_picker=True, start_dir=start_dir, callback=callback, allow_multiple=allow_multiple, custom_popup_flags=custom_popup_flags)

