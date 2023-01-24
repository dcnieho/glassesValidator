from imgui_bundle import imgui, immapp, hello_imgui, glfw_window_hello_imgui
import glfw
import time
import threading
import numpy as np
import OpenGL.GL as gl

# simple GUI provider for viewer and coder windows in glassesValidator.process

class GUI:
    def __init__(self):
        self._should_exit = False
        self._thread = None
        self._frame_nr = 0
        self._frame = None
        self._glfw_window = None

        self._draw_callback = None

    def __del__(self):
        self.stop()

    def start(self, window_name):
        # optional sizing info, will scale and position window up to that
        # but still fitting on current monitor
        # take into account DPI scale
        if self._thread is not None:
            raise RuntimeError('The gui is already running, cannot start again')
        self._thread = threading.Thread(target=self._thread_start_fun, args=(window_name,))
        self._thread.start()

    def get_state(self):
        return (self._user_closed_window,)

    def stop(self):
        self._should_exit = True
        if self._thread is not None:
            self._thread.join()
        self._thread = None

    def update_image(self, frame, pts = None, frame_nr = None):
        # since this has an independently running loop,
        # need to update image whenever new one available
        self._frame = frame # just copy ref is enough
        self._frame_pts = pts
        if frame_nr:
            self._frame_nr = frame_nr
        else:
            self._frame_nr += 1

    def register_draw_callback(self, callback):
        # e.g. for drawing overlays
        self._draw_callback = callback

    def _thread_start_fun(self, window_name):
        self._last_frame = None
        self._last_frame_nr = 0
        self._frame_pts = None
        self._lastT=0.
        self._texID=None
        self._should_exit = False
        self._hidden = False
        self._user_closed_window = False
        self._dpi_fac = 1

        def post_init():
            def close_callback(window: glfw._GLFWwindow):
                self._user_closed_window = True

            glfw.swap_interval(0)
            self._glfw_window = glfw_window_hello_imgui()
            self._dpi_fac = hello_imgui.dpi_window_size_factor()
            glfw.hide_window(self._glfw_window)
            self._hidden = True
            glfw.set_window_close_callback(self._glfw_window, close_callback)

        params = hello_imgui.RunnerParams()
        params.app_window_params.window_geometry.size_auto = True
        params.app_window_params.restore_previous_geometry = False
        params.app_window_params.window_title = window_name
        params.fps_idling.fps_idle = 0
        params.callbacks.show_gui = self._draw_gui
        params.callbacks.post_init = post_init

        immapp.run(params)

    def _draw_gui(self):
        # check if we should exit
        if self._should_exit:
            # clean up
            if self._texID:
                # delete
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                gl.glDeleteTextures(1, [self._texID])
                self._texID = None
            # and kill
            hello_imgui.get_runner_params().app_shall_exit = True
            # nothing more to do
            return

        # manual vsync with a sleep, so that other thread can run
        # thats crappy vsync, but ok for our purposes
        thisT = time.perf_counter()
        elapsedT = thisT-self._lastT
        self._lastT = thisT

        if elapsedT < 1/60:
            time.sleep(1/60-elapsedT)

        # upload texture if needed
        if self._texID is None:
            self._texID = gl.glGenTextures(1)
        if self._frame_nr != self._last_frame_nr:
            # if first time we're showing something
            if self._last_frame is None:
                # tell window to resize
                hello_imgui.get_runner_params().app_window_params.window_geometry.resize_app_window_at_next_frame = True
                # and show window if needed
                if self._hidden:
                    glfw.show_window(self._glfw_window)
                    self._hidden = False
            # upload texture
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._texID)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, self._frame.shape[1], self._frame.shape[0], 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, self._frame)
            self._last_frame_nr = self._frame_nr
            self._last_frame = self._frame

        # draw image
        if self._last_frame is not None:
            imgui.image(self._texID, imgui.ImVec2(self._last_frame.shape[1]*self._dpi_fac, self._last_frame.shape[0]*self._dpi_fac))

        # draw bottom status overlay
        ws = imgui.get_window_size()
        ts = imgui.calc_text_size('')
        imgui.set_cursor_pos_y(ws[1]-ts[1])
        imgui.push_style_color(imgui.Col_.child_bg, ( 0.0, 0.0, 0.0, 0.4 ) )
        imgui.begin_child("##status_overlay", size=(-imgui.FLT_MIN,ts[1]))
        if (self._frame_pts):
            imgui.text(" %8.3f [%d]" % (self._frame_pts, self._last_frame_nr))
        else:
            imgui.text(" %d" % (self._last_frame_nr,))
        imgui.end_child()
        imgui.pop_style_color()

        if self._draw_callback:
            self._draw_callback()