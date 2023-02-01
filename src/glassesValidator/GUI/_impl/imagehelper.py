# edited from https://gist.github.com/Willy-JL/9c5116e5a11abd559c56f23aa1270de9
from PIL import Image, ImageSequence, UnidentifiedImageError
import pathlib
from imgui_bundle import imgui
import OpenGL.GL as gl
import numpy


class ImageHelper:
    def __init__(self, path: str | pathlib.Path, glob=""):
        self.width = 1
        self.height = 1
        self.glob = glob
        self.loaded = False
        self.loading = False
        self.applied = False
        self.missing = False
        self.invalid = False
        self.frame_count = 1
        self.animated = False
        self.prev_time = 0.0
        self.current_frame = -1
        self.frame_elapsed = 0.0
        self.data: bytes | list[bytes] = None
        self._texture_id: numpy.uint32 = None
        self.frame_durations: list[float] = None
        self.path = pathlib.Path(path)
        self.resolved_path: pathlib.Path = None
        self.resolve()

    def reset(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 0, 0, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, b"\x00\x00\x00\xff")
        self.applied = False

    @staticmethod
    def get_rgba_pixels(image: Image.Image):
        if image.mode == "RGB":
            return image.tobytes("raw", "RGBX")
        else:
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            return image.tobytes("raw", "RGBA")

    def set_missing(self):
        self.missing = True
        self.loaded = True
        self.loading = False

    def resolve(self):
        self.resolved_path = self.path
        if self.glob:
            paths = list(self.resolved_path.glob(self.glob))
            if not paths:
                self.set_missing()
                return
            # If you want you can setup preferred extensions like this:
            paths.sort(key=lambda path: path.suffix != ".gif")
            # This will prefer .gif files!
            self.resolved_path = paths[0]
        if self.resolved_path.is_file():
            self.missing = False
        else:
            self.set_missing()
            return

    def reload(self):
        self.resolve()
        if self.missing:
            return
        try:
            image = Image.open(self.resolved_path)
            self.invalid = False
        except UnidentifiedImageError:
            self.invalid = True
            self.loaded = True
            self.loading = False
            return
        self.width, self.height = image.size
        if hasattr(image, "n_frames") and image.n_frames > 1:
            self.animated = True
            self.data = []
            self.frame_durations = []
            for frame in ImageSequence.Iterator(image):
                self.data.append(self.get_rgba_pixels(frame))
                if (duration := image.info["duration"]) < 1:
                    duration = 100
                self.frame_durations.append(duration / 1250)
                # Technically this should be / 1000 (millis to seconds) but I found that 1250 works better...
            self.frame_count = len(self.data)
        else:
            self.data = self.get_rgba_pixels(image)
        self.loaded = True
        self.loading = False

    def apply(self, data: bytes):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.width, self.height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)

    @property
    def texture_id(self):
        if self._texture_id is None:
            self._texture_id = gl.glGenTextures(1)
        if not self.loaded:
            if not self.loading:
                self.loading = True
                self.reset()
                self.reload()
        elif not self.missing:
            if self.animated:
                if self.prev_time != (new_time := imgui.get_time()):
                    self.prev_time = new_time
                    self.frame_elapsed += imgui.get_io().delta_time
                    while (excess := self.frame_elapsed - self.frame_durations[max(self.current_frame, 0)]) > 0:
                        self.frame_elapsed = excess
                        self.applied = False
                        self.current_frame += 1
                        if self.current_frame == self.frame_count:
                            self.current_frame = 0
                if not self.applied:
                    self.apply(self.data[self.current_frame])
                    self.applied = True
            elif not self.applied:
                self.apply(self.data)
                self.applied = True
        return self._texture_id

    def render(self, width: int, height: int, *args, **kwargs):
        if self.missing:
            return False
        if imgui.is_rect_visible((width, height)):
            if "rounding" in kwargs:
                flags = kwargs.pop("flags", None)
                if flags is None:
                    flags = imgui.ImDrawFlags_.round_corners_all
                pos = imgui.get_cursor_screen_pos()
                pos2 = (pos.x + width, pos.y + height)
                draw_list = imgui.get_window_draw_list()
                draw_list.add_image_rounded(self.texture_id, pos, pos2, (0,0), (1,1), col=0xffffffff, *args, flags=flags, **kwargs)
                imgui.dummy((width, height))
            else:
                imgui.image(self.texture_id, width, height, *args, **kwargs)
            return True
        else:
            # Skip if outside view
            imgui.dummy((width, height))
            return False

    def crop_to_ratio(self, ratio: int | float, fit=False):
        img_ratio = self.width / self.height
        if (img_ratio >= ratio) != fit:
            crop_h = self.height
            crop_w = crop_h * ratio
            crop_x = (self.width - crop_w) / 2
            crop_y = 0
            left = crop_x / self.width
            top = 0
            right = (crop_x + crop_w) / self.width
            bottom = 1
        else:
            crop_w = self.width
            crop_h = crop_w / ratio
            crop_y = (self.height - crop_h) / 2
            crop_x = 0
            left = 0
            top = crop_y / self.height
            right = 1
            bottom = (crop_y + crop_h) / self.height
        return (left, top), (right, bottom)