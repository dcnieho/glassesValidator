import OpenGL.GL as gl
import pathlib
import functools
import traceback
import typing
import random
import imgui
import math
import glfw
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




# https://github.com/pyimgui/pyimgui/blob/24219a8d4338b6e197fa22af97f5f06d3b1fe9f7/doc/examples/integrations_glfw3.py
def impl_glfw_init(width: int, height: int, window_name: str):
    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    if globals.os is Os.MacOS:
        # OS X supports only forward-compatible core profiles from 3.2
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(width, height, window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    return window


def validate_geometry(x, y, width, height):
    window_pos = (x, y)
    window_size = (width, height)
    valid = False
    for monitor in glfw.get_monitors():
        monitor_area = glfw.get_monitor_workarea(monitor)
        monitor_pos = (monitor_area[0], monitor_area[1])
        monitor_size = (monitor_area[2], monitor_area[3])
        # Horizontal check, at least 1 pixel on x axis must be in monitor
        if (window_pos[0]) >= (monitor_pos[0] + monitor_size[0]):
            continue  # Too right
        if (window_pos[0] + window_size[0]) <= (monitor_pos[0]):
            continue  # Too left
        # Vertical check, at least the pixel above window must be in monitor (titlebar)
        if (window_pos[1] - 1) >= (monitor_pos[1] + monitor_size[1]):
            continue  # Too low
        if (window_pos[1]) <= (monitor_pos[1]):
            continue  # Too high
        valid = True
        break
    return valid

def get_current_monitor(wx, wy, ww, wh):
    import ctypes
    # so we always return something sensible
    monitor = glfw.get_primary_monitor()
    bestoverlap = 0
    for mon in glfw.get_monitors():
        monitor_area = glfw.get_monitor_workarea(mon)
        mx, my = monitor_area[0], monitor_area[1]
        mw, mh = monitor_area[2], monitor_area[3]

        overlap = \
            max(0, min(wx + ww, mx + mw) - max(wx, mx)) * \
            max(0, min(wy + wh, my + mh) - max(wy, my))
        
        if bestoverlap < overlap:
            bestoverlap = overlap
            monitor = mon

    return monitor, ctypes.cast(ctypes.pointer(monitor), ctypes.POINTER(ctypes.c_long)).contents.value


def push_disabled(block_interaction=True):
    if block_interaction:
        imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
    imgui.push_style_var(imgui.STYLE_ALPHA, imgui.style.alpha *  0.5)


def pop_disabled(block_interaction=True):
    if block_interaction:
        imgui.internal.pop_item_flag()
    imgui.pop_style_var()


def center_next_window():
    size = imgui.io.display_size
    imgui.set_next_window_position(size.x / 2, size.y / 2, pivot_x=0.5, pivot_y=0.5)


def constrain_next_window():
    size = imgui.io.display_size
    imgui.set_next_window_size_constraints((0, 0), (size.x * 0.9, size.y * 0.9))


def close_weak_popup():
    if not imgui.is_popup_open("", imgui.POPUP_ANY_POPUP_ID):
        # This is the topmost popup
        if imgui.io.keys_down[glfw.KEY_ESCAPE]:
            # Escape is pressed
            imgui.close_current_popup()
            return True
        elif imgui.is_mouse_clicked():
            # Mouse was just clicked
            pos = imgui.get_window_position()
            size = imgui.get_window_size()
            if not imgui.is_mouse_hovering_rect(pos.x, pos.y, pos.x + size.x, pos.y + size.y, clip=False):
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
            new_pos_x = cur_pos_x + imgui.get_content_region_available_width() - btns_width
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
            args[1] = args[1] + "###popup_" + rand_num_str()
        popup_func = functools.partial(*args, **kwargs)
    else:
        popup_func = args[0]
    if bottom:
        globals.popup_stack.insert(0, popup_func)
    else:
        globals.popup_stack.append(popup_func)
    return popup_func


def set_all(input: dict[int, bool], value, subset: list[int] = None):
    if subset is not None:
        for r in subset:
            input[r] = value
    else:
        for r in input:
            input[r] = value

def calc_circle_auto_segment_count(radius, max_error=0.3):
    # see IM_DRAWLIST_CIRCLE_AUTO_SEGMENT_CALC in imgui_interal.h
    IM_ROUNDUP_TO_EVEN                  = lambda x: math.ceil(x / 2.) * 2
    IM_DRAWLIST_CIRCLE_AUTO_SEGMENT_MIN = 4
    IM_DRAWLIST_CIRCLE_AUTO_SEGMENT_MAX = 512
    return max(min(IM_ROUNDUP_TO_EVEN(math.pi / math.acos(1 - min([max_error, radius]) / radius)), IM_DRAWLIST_CIRCLE_AUTO_SEGMENT_MAX), IM_DRAWLIST_CIRCLE_AUTO_SEGMENT_MIN)

def draw_spinner(label: str, radius1: float, radius2: float, radius3: float, thickness: float, c1 = 0xffffffff, c2 = 0x80ffffff, c3 = 0xffffffff, speed = 2.8, angle = math.pi):    # NB: ABGR order for colors when specified like this
    # based on ImSpinner::SpinnerAngTriple from https://github.com/dalerank/imspinner
    # but implemented a bit roundabout because a whole bunch of needed imgui and imgui.internal functions are not exposed by pyimgui

    # determine size
    style  = imgui.get_style()
    radii  = [radius1, radius2, radius3]
    radius = max(radii)
    size   = [2*radius, 2*(radius+style.frame_padding.y)]

    # draw dummy of that size and reset cursor back to position before if that dummy was visible. The dummy reserves the space we need to draw in
    cur_pos = imgui.get_cursor_pos()
    imgui.push_id(label)
    imgui.dummy(size[0],size[1])
    imgui.pop_id()
    imgui.set_cursor_pos(cur_pos)

    # if our draw area is visible, draw the spinner
    if imgui.is_item_visible():
        pos = imgui.get_cursor_screen_pos()
        center = [x+y/2 for x,y in zip(pos,size)]
        num_segments = calc_circle_auto_segment_count(radius) * 2
        angle_offset = angle / num_segments
        colors = [c1, c2, c3]
        fac = [1, 1.2, 0.9]

        t = imgui.get_time()
        draw_list = imgui.get_window_draw_list()
        for i,r in enumerate(radii):
            path = []
            start = fac[i] * t * speed
            neg   = -1 if i==1 else 1
            for k in range(num_segments):
                a = start + (k * angle_offset)
                path.append((center[0] + math.cos(neg*a) * r, center[1] + math.sin(neg*a) * r))

            draw_list.add_polyline(path, colors[i], flags=imgui.DRAW_NONE, thickness=thickness)

def bounce_dots(label: str, thickness: float, color = 0xffffffff, speed = 2.8, dots=3):
    # based on ImSpinner::SpinnerBounceDots from https://github.com/dalerank/imspinner
    # but implemented a bit roundabout because a whole bunch of needed imgui and imgui.internal functions are not exposed by pyimgui
    nextItemKoeff = 2.5
    heightKoeff = 2.
    heightSpeed = 0.8

    style  = imgui.get_style()
    size   = [(thickness * nextItemKoeff) * dots + style.frame_padding.x, thickness * 4 * heightKoeff + style.frame_padding.y]

    # draw dummy of that size and reset cursor back to position before if that dummy was visible. The dummy reserves the space we need to draw in
    cur_pos = imgui.get_cursor_pos()
    imgui.push_id(label)
    imgui.dummy(size[0],size[1])
    imgui.pop_id()
    imgui.set_cursor_pos(cur_pos)

    # if our draw area is visible, draw the spinner
    if imgui.is_item_visible():
        pos = imgui.get_cursor_screen_pos()
        center = [x+y/2 for x,y in zip(pos,size)]
        num_segments = calc_circle_auto_segment_count(thickness) * 2

        start = imgui.get_time() * speed;
        draw_list = imgui.get_window_draw_list()
        offset = math.pi / dots
        
        for i in range(dots):
            a = start + (math.pi - i * offset)
            sina = math.sin(a * heightSpeed)
            y = center[1] + sina * thickness * heightKoeff
            if (y > center[1]):
                y = center[1]
            y+=thickness * heightKoeff  # move down so animation is centered around center
            draw_list.add_circle_filled(pos.x + style.frame_padding.x  + i * (thickness * nextItemKoeff), y, thickness, color, num_segments)