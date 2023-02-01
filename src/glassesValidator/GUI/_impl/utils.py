import pathlib
import functools
import traceback
import typing
import random
from imgui_bundle import imgui
import sys
import os

from . import globals, msgbox
from .structs import Os




def rand_num_str(len=8):
    return "".join((random.choice('0123456789') for _ in range(len)))


def get_data_path():
    if globals.project_path is not None:
        return globals.project_path
    else:
        return globals.data_path


def is_project_folder(folder: str | pathlib.Path):
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        return False
    # a project directory should contain the (empty)
    # glassesValidator.project file and imgui.ini file
    return (folder/'imgui.ini').is_file() and (folder/'glassesValidator.project').is_file()


def init_project_folder(folder: str | pathlib.Path, imgui_ini_saver: typing.Callable = None):
    folder = pathlib.Path(folder)
    if not folder.is_dir():
        return
    # a project directory should contain the empty
    # glassesValidator.project file, so we create it
    # here.
    # Also, a copy of imgui.ini to persist some settings
    # of the current shown GUI is good to have, so save it here.

    if imgui_ini_saver is not None:
        imgui_ini_saver(folder/'imgui.ini')
    else:
        imgui.save_ini_settings_to_disk(str(folder/'imgui.ini'))
    (folder/'glassesValidator.project').touch()


def fast_scandir(dirname):
    if not dirname.is_dir():
        return []
    subfolders= [pathlib.Path(f.path) for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders


# https://gist.github.com/Willy-JL/f733c960c6b0d2284bcbee0316f88878
def get_traceback(*exc_info: list):
    exc_info = exc_info or sys.exc_info()
    tb_lines = traceback.format_exception(*exc_info)
    tb = "".join(tb_lines)
    return tb




def push_disabled(block_interaction=True):
    if block_interaction:
        imgui.internal.push_item_flag(imgui.internal.ItemFlags_.disabled, True)
    imgui.push_style_var(imgui.StyleVar_.alpha, imgui.get_style().alpha *  0.5)


def pop_disabled(block_interaction=True):
    if block_interaction:
        imgui.internal.pop_item_flag()
    imgui.pop_style_var()


def center_next_window():
    pos = imgui.get_window_pos()
    size = imgui.get_window_size()
    imgui.set_next_window_pos((pos.x+size.x/2, pos.y+size.y/2), pivot=(0.5,0.5), cond=imgui.Cond_.appearing)

def constrain_next_window():
    size = imgui.io.display_size
    imgui.set_next_window_size_constraints((0, 0), (size.x * 0.9, size.y * 0.9))

def close_weak_popup():
    if not imgui.is_popup_open("", imgui.PopupFlags_.any_popup_id):
        # This is the topmost popup
        if imgui.is_key_pressed(imgui.Key.escape):
            # Escape is pressed
            imgui.close_current_popup()
            return True
        elif imgui.is_mouse_clicked(imgui.MouseButton_.left):
            # Mouse was just clicked
            pos = imgui.get_window_pos()
            size = imgui.get_window_size()
            if not imgui.is_mouse_hovering_rect(pos, (pos.x+size.x, pos.y+size.y), clip=False):
                # Popup is not hovered
                imgui.close_current_popup()
                return True
    return False

def popup(label: str, popup_content: typing.Callable, buttons: dict[str, typing.Callable] = None, closable=True, outside=True):
    if buttons is True:
        buttons = {
            "󰄬 Ok": None
        }
    if not imgui.is_popup_open(label):
        imgui.open_popup(label)
    closed = False
    opened = 1
    constrain_next_window()
    center_next_window()
    if imgui.begin_popup_modal(label, closable or None, flags=globals.gui.popup_flags)[0]:
        if outside:
            closed = close_weak_popup()
        imgui.begin_group()
        popup_content()
        imgui.end_group()
        imgui.spacing()
        if buttons:
            btns_width = sum(imgui.calc_text_size(name).x for name in buttons) + (2 * len(buttons) * imgui.style.frame_padding.x) + (imgui.style.item_spacing.x * (len(buttons) - 1))
            cur_pos_x = imgui.get_cursor_pos_x()
            new_pos_x = cur_pos_x + imgui.get_content_region_avail().x - btns_width
            if new_pos_x > cur_pos_x:
                imgui.set_cursor_pos_x(new_pos_x)
            for label, callback in buttons.items():
                if imgui.button(label):
                    if callback:
                        callback()
                    imgui.close_current_popup()
                    closed = True
                imgui.same_line()
    else:
        opened = 0
        closed = True
    return opened, closed

def push_popup(*args, bottom=False, **kwargs):
    if len(args) + len(kwargs) > 1:
        if args[0] is popup or args[0] is msgbox.msgbox:
            args = list(args)
            args[1] = args[1] + "##popup_" + rand_num_str()
        popup_func = functools.partial(*args, **kwargs)
    else:
        popup_func = args[0]
    if bottom:
        globals.popup_stack.insert(0, popup_func)
    else:
        globals.popup_stack.append(popup_func)
    return popup_func


def set_all(input: dict[int, bool], value, subset: list[int] = None, predicate: typing.Callable = None):
    if subset is None:
        subset = (r for r in input)
    for r in subset:
        if not predicate or predicate(r):
            input[r] = value

def selectable_item_logic(id: int, selected: dict[int,typing.Any], last_clicked_id: int, sorted_ids: list[int],
                          selectable_clicked: bool, new_selectable_state: bool,
                          allow_multiple=True, overlayed_hovered=False, overlayed_clicked=False, new_overlayed_state=False):
    if overlayed_clicked:
        if not allow_multiple:
            set_all(selected, False)
        selected[id] = new_overlayed_state
        last_clicked_id = id
    elif selectable_clicked and not overlayed_hovered: # don't enter this branch if interaction is with another overlaid actionable item
        if not allow_multiple:
            set_all(selected, False)
            selected[id] = new_selectable_state
        else:
            num_selected = sum([selected[id] for id in sorted_ids])
            if not imgui.io.key_ctrl:
                # deselect all, below we'll either select all, or range between last and current clicked
                set_all(selected, False)

            if imgui.io.key_shift:
                # select range between last clicked and just clicked item
                idx              = sorted_ids.index(id)
                last_clicked_idx = sorted_ids.index(last_clicked_id)
                idxs = sorted([idx, last_clicked_idx])
                for rid in range(idxs[0],idxs[1]+1):
                    selected[sorted_ids[rid]] = True
            else:
                selected[id] = True if num_selected>1 and not imgui.io.key_ctrl else new_selectable_state

            # consistent with Windows behavior, only update last clicked when shift not pressed
            if not imgui.io.key_shift:
                last_clicked_id = id

    return last_clicked_id
