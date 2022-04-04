import numpy as np
import PySimpleGUI as sg
import cv2

def array_to_data(array):
    from PIL import Image
    from io import BytesIO
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data

class GUIHandler():
    def __init__(self, scale = 1.0, page = 0, **kwargs):
        self.scale = scale
        self.page = page

    def frame_transform(self, image):
        return image
    
    def handle_image_change(self, gui):
        (ww, wh) = gui.window.Size
        img = gui.images[self.page]
        (h, w, *_) = img.shape
        w_prop = ((ww / 2) - 20) / w
        h_prop = (wh - 50) / h
        norm_scale = self.scale * min(w_prop, h_prop)
        w = int(w*norm_scale)
        h = int(h*norm_scale)

        transformed_img, *orig = self.frame_transform(img)
        if len(orig) > 0:
            img = orig[0]
        img = cv2.resize(img, fx = norm_scale, fy = norm_scale, dsize = (w, h))
        transformed_img = cv2.resize(transformed_img, fx = norm_scale, fy = norm_scale, dsize = (w, h))
        gui.window['img'].update(array_to_data(img))
        gui.window['img'].set_size((w, h))
        gui.window['img2'].update(array_to_data(transformed_img))
        gui.window['img2'].set_size((w, h))
        gui.window['label'].update(value = f'Page {self.page + 1} de {len(gui.images)}')
    def handle_tick(self, gui, value):
        self.handle_image_change(gui)
    def handle_init(self, gui, value):
        pass
    def handle_X(self, gui, value):
        return True
    def handle_None(self, gui, value):
        return True
    def handle_more_zoom(self, gui, value):
        self.scale += 0.1
        self.handle_image_change(gui)
    def handle_less_zoom(self, gui, value):
        self.scale -= 0.1
        self.handle_image_change(gui)
    def handle_prev(self, gui, value):
        if self.page > 0:
            self.page -= 1
            self.handle_image_change(gui)
    def handle_next(self, gui, value):
        if self.page < len(gui.images) - 1:
            self.page += 1
            self.handle_image_change(gui)
    def handle_window_event(self, gui, value):
        self.handle_image_change(gui)

    @property
    def buttons(self):
        return [
            sg.Button("<", key = 'prev'),
            sg.Button("+", key = 'more_zoom'),
            sg.Button("X"),
            sg.Button("-", key = 'less_zoom'),
            sg.Button(">", key = 'next'),
            sg.Text(key = 'label')
        ]


class GUI():
    def layout(self):
        return [
            self.handler.buttons,
            [ sg.Image(key = 'img'), sg.Image(key = 'img2') ]
        ]

    def __init__(self, images = [], handler = GUIHandler()):
        self.handler = handler
        self.images = list(map(np.array, images))
        self.window = sg.Window('window', self.layout(), resizable = True, finalize = True, element_justification = 'c')
        self.window.bind('<Configure>', 'window_event')
        self.emit('tick')
    def emit(self, name = 'tick', value = None):
        self.window.write_event_value(name, value)
    def tick(self, timeout = None):
        event, values = self.window.read(timeout = timeout)
        handler = None
        try:
            handler = self.handler.__getattribute__(f'handle_{event}')
            ret = handler(self, values)
            if ret is not None:
                return False
        except AttributeError:
            pass
        print("GUI event: ", event, handler is not None, values)
        return True

    def run(self):
        while self.tick():
            pass
        self.window.close()
