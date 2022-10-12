# https://gist.github.com/Willy-JL/82137493896d385a74d148534691b6e1
import pathlib
import typing
import string
import imgui
import glfw
import sys
import os
import dataclasses
import natsort

from . import globals, utils
from .structs import SortSpec

@dataclasses.dataclass
class DirEntry:
    name: str
    is_dir: bool
    full_path: pathlib.Path

class FilePicker:
    flags = (
        imgui.WINDOW_NO_MOVE |
        imgui.WINDOW_NO_RESIZE |
        imgui.WINDOW_NO_COLLAPSE |
        imgui.WINDOW_NO_SAVED_SETTINGS |
        imgui.WINDOW_ALWAYS_AUTO_RESIZE
    )

    def __init__(self, title="File picker", dir_picker=False, start_dir: str | pathlib.Path = None, callback: typing.Callable = None, allow_multiple = True, custom_popup_flags=0):
        self.title = title
        self.active = True
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
                        self.items[i] = DirEntry(item.name,item.is_dir(),item)
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
        if not self.active:
            return 0, True
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
        size = io.display_size
        imgui.set_next_window_position(size.x / 2, size.y / 2, pivot_x=0.5, pivot_y=0.5)
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
            confirmed, dir = imgui.input_text("##location_bar", str(self.dir), flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
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
            imgui.begin_child("##folder_contents", height=size.y * 0.65, width=imgui.get_item_rect_size().x)
            if self.msg:
                imgui.text_unformatted(self.msg)
            else:
                table_flags = (
                    imgui.TABLE_SCROLL_Y |
                    imgui.TABLE_SCROLL_X |
                    imgui.TABLE_SORTABLE |
                    imgui.TABLE_SORT_MULTI |
                    imgui.TABLE_ROW_BACKGROUND |
                    imgui.TABLE_SIZING_FIXED_FIT |
                    imgui.TABLE_NO_HOST_EXTEND_Y
                )
                if imgui.begin_table(f"##folder_list",column=2,flags=table_flags):
                    frame_height = imgui.get_frame_height()

                    # Setup
                    checkbox_width = frame_height
                    imgui.table_setup_column("󰄵 Selector", imgui.TABLE_COLUMN_NO_HIDE | imgui.TABLE_COLUMN_NO_SORT | imgui.TABLE_COLUMN_NO_RESIZE | imgui.TABLE_COLUMN_NO_REORDER, init_width_or_weight=checkbox_width)  # 0
                    imgui.table_setup_column("󰌖 Name", imgui.TABLE_COLUMN_DEFAULT_SORT | imgui.TABLE_COLUMN_NO_HIDE | imgui.TABLE_COLUMN_NO_RESIZE)  # 3
                    imgui.table_setup_scroll_freeze(1, 1)  # Sticky column headers and selector row

                    sort_specs = imgui.table_get_sort_specs()
                    self.sort_items(sort_specs)

                    # Headers
                    imgui.table_next_row(imgui.TABLE_ROW_HEADERS)
                    imgui.table_set_column_index(0) # checkbox column: reflects whether all, some or none of visible recordings are selected, and allows selecting all or none
                    # get state
                    num_selected = sum([self.selected[id] for id in self.selected])
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
                        imgui.internal.push_item_flag(imgui.internal.ITEM_MIXED_VALUE,True)
                    clicked, new_state = imgui.checkbox(f"##header_checkbox", multi_selected_state==1, frame_size=(0,0), do_vertical_align=False)
                    if multi_selected_state==0:
                        imgui.internal.pop_item_flag()

                    imgui.table_set_column_index(1)
                    imgui.table_header(imgui.table_get_column_name(1)[2:])

                    if clicked:
                        utils.set_all(self.selected, new_state, subset = self.sorted_items, predicate=self.predicate)
                    
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
                        checkbox_clicked, checkbox_hovered = False, False
                        has_drawn_hitbox = False

                        disable_item = self.predicate and not self.predicate(id)
                        if disable_item:
                            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha *  0.5)

                        for ci in range(2):
                            if not (imgui.table_get_column_flags(ci) & imgui.TABLE_COLUMN_IS_ENABLED):
                                continue
                            imgui.table_set_column_index(ci)

                            # Row hitbox
                            if not has_drawn_hitbox:
                                # hitbox needs to be drawn before anything else on the row so that, together with imgui.set_item_allow_overlap(), hovering button
                                # or checkbox on the row will still be correctly lead detected.
                                # this is super finicky, but works. The below together with using a height of frame_height+cell_padding_y
                                # makes the table row only cell_padding_y/2 longer. The whole row is highlighted correctly
                                cell_padding_y = imgui.style.cell_padding.y
                                cur_pos_y = imgui.get_cursor_pos_y()
                                imgui.set_cursor_pos_y(cur_pos_y - cell_padding_y/2)
                                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 0.)
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0.,0.))
                                imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (0.,cell_padding_y))
                                # make selectable completely transparent
                                imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0., 0., 0., 0.)
                                imgui.push_style_color(imgui.COLOR_HEADER       , 0., 0., 0., 0.)
                                imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0., 0., 0., 0.)
                                selectable_clicked, selectable_out = imgui.selectable(f"##{id}_hitbox", self.selected[id], flags=imgui.SELECTABLE_SPAN_ALL_COLUMNS|imgui.internal.SELECTABLE_SELECT_ON_CLICK, height=frame_height+cell_padding_y)
                                # instead override table row background color
                                if selectable_out:
                                    imgui.table_set_background_color(imgui.TABLE_BACKGROUND_TARGET_ROW_BG0, imgui.color_convert_float4_to_u32(*style_selected_row))
                                elif imgui.is_item_hovered():
                                    imgui.table_set_background_color(imgui.TABLE_BACKGROUND_TARGET_ROW_BG0, imgui.color_convert_float4_to_u32(*style_hovered_row))
                                imgui.set_cursor_pos_y(cur_pos_y)   # instead of imgui.same_line(), we just need this part of its effect
                                imgui.set_item_allow_overlap()
                                imgui.pop_style_color(3)
                                imgui.pop_style_var(3)
                                has_drawn_hitbox = True
                        
                            if ci==1:
                                # (Invisible) button because it aligns the following draw calls to center vertically
                                imgui.push_style_var(imgui.STYLE_FRAME_BORDERSIZE, 0.)
                                imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0.,imgui.style.frame_padding.y))
                                imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (0.,imgui.style.item_spacing.y))
                                imgui.push_style_color(imgui.COLOR_BUTTON, 0.,0.,0.,0.)
                                imgui.button(f"##{id}_id", width=imgui.FLOAT_MIN)
                                imgui.pop_style_color()
                                imgui.pop_style_var(3)
                        
                                imgui.same_line()

                            match ci:
                                case 0:
                                    # Selector
                                    checkbox_clicked, checkbox_out = imgui.checkbox(f"##{id}_selected", self.selected[id], frame_size=(0,0))
                                    checkbox_hovered = imgui.is_item_hovered()
                                case 1:
                                    # Name
                                    prefix = self.dir_icon if self.items[id].is_dir else self.file_icon
                                    imgui.text(prefix+self.items[id].name)
                                    
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
                            if not imgui.io.key_ctrl and not imgui.io.key_shift and imgui.is_mouse_double_clicked():
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
            disable_ok = not num_selected and not self.dir_picker
            if disable_ok:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha *  0.5)
            if imgui.button("󰄬 Ok"):
                imgui.close_current_popup()
                closed = True
            if disable_ok:
                imgui.internal.pop_item_flag()
                imgui.pop_style_var()
            # Selected text
            imgui.same_line()
            if self.dir_picker and not num_selected:
                imgui.text(f"  Selected the current directory ({self.dir.name})")
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
            self.active = False
        return opened, closed

    def sort_items(self, sort_specs_in: imgui.core._ImGuiTableSortSpecs):
        if sort_specs_in.specs_dirty or self.require_sort:
            ids = list(self.items)
            for sort_spec in sort_specs_in.specs:
                sort_spec = SortSpec(index=sort_spec.column_index, reverse=bool(sort_spec.sort_direction - 1))
                match sort_spec.index:
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

