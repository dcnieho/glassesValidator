import sys
from PIL import Image
from imgui_bundle import imgui
import glfw
import OpenGL.GL as gl
import importlib.resources
from ._impl import imagehelper


def glfw_error_callback(error: int, description: str):
    sys.stderr.write(f"Glfw Error {error}: {description}\n")

def glfw_init(width: int, height: int, window_name: str, hide = False) -> glfw._GLFWwindow:
    glfw.set_error_callback(glfw_error_callback)
    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    if sys.platform.startswith("darwin"):
        # OS X supports only forward-compatible core profiles from 3.2
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        if hide:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(width, height, window_name, None, None)
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    glfw.make_context_current(window)
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

icon_texture = None
def setup_glfw_window(name: str, size = (300,300), pos = None, hide_main_window = False):
    window = glfw_init(*size, name, hide_main_window)
    if size and pos:
        if all([isinstance(x, int) for x in pos]) and len(pos) == 2 and validate_geometry(*pos, *size):
            glfw.set_window_pos(window, *pos)
    screen_pos = glfw.get_window_pos(window)
    screen_size= glfw.get_window_size(window)

    icon_path = importlib.resources.files('glassesValidator.resources.icons') / 'icon.png'
    with importlib.resources.as_file(icon_path) as icon_file:
        global icon_texture
        icon_texture = imagehelper.ImageHelper(icon_file)
        icon_texture.reload()
        glfw.set_window_icon(window, 1, Image.open(icon_file))

    return window, screen_pos, screen_size

def get_monitor_scaling(screen_pos, screen_size):
    # determine what monitor we're (mostly) on
    mon, monitor = get_current_monitor(*screen_pos, *screen_size)
    # get scaling of that monitor
    if sys.platform.startswith("darwin"):
        xscale, yscale = 1., 1.
    else:
        xscale, yscale = glfw.get_monitor_content_scale(mon)
    return monitor, max(xscale, yscale)

def setup_imgui():
    imgui.create_context()
    imgui.io = imgui.get_io()
    imgui.io.config_drag_click_to_input_text = True

def setup_imgui_impl(window, vsync_ratio):
    # transfer the window address to imgui.backends.glfw_init_for_open_gl
    import ctypes
    window_address = ctypes.cast(window, ctypes.c_void_p).value
    imgui.backends.glfw_init_for_open_gl(window_address, True)
    imgui.backends.opengl3_init("#version 150")
    glfw.swap_interval(vsync_ratio)


def destroy_imgui_glfw():
    imgui.backends.opengl3_shutdown()
    imgui.backends.glfw_shutdown()
    if (ctx := imgui.get_current_context()) is not None:
        imgui.io.set_ini_filename('')   # don't store settings to ini, user needs to do that themselves if wanted
        imgui.destroy_context(ctx)
    glfw.terminate()