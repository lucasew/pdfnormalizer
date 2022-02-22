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

class GUIWithImageTransformer():
    scale = 1.0
    page = 0
    buttonbar = [
        sg.Button("<", key = 'prev'),
        sg.Button("+", key = 'more_zoom'),
        sg.Button("X"),
        sg.Button("-", key = 'less_zoom'),
        sg.Button(">", key = 'next'),
        sg.Text(key = 'label')
    ]

    def _get_button_bar(self):
        buttons = []
        try:
            super_buttons = super()._get_button_bar()
            for button in super_buttons:
                buttons.append(button)
        except AttributeError:
            pass
        for button in self.buttonbar:
            buttons.append(button)
        return buttons

    @property
    def layout(self):
        buttonbar = self._get_button_bar()
        return [
            buttonbar,
            [ sg.Image(key = 'img'), sg.Image(key = 'img2') ]
        ]

    def __init__(self, images = [], start_page = 0, scale = 1.0):
        import numpy as np
        self.scale = scale
        self.page = start_page
        self.images = list(map(np.array, images))
        self.window = sg.Window('window', self.layout, resizable = True, finalize = True, element_justification = 'c')
        self.window.bind('<Configure>', 'Configure')
        self.window.write_event_value('init', None)
    def handle_X(self, value):
        return True
    def handle_None(self, value):
        return True
    def handle_more_zoom(self, value):
        self.scale += 0.1
    def handle_less_zoom(self, value):
        self.scale -= 0.1
    def handle_prev(self, value):
        if self.page > 0:
            self.page -= 1
    def handle_next(self, value):
        if self.page < len(self.images) - 1:
            self.page += 1
    def frame_transform(self, frame):
        return frame
    def run(self):
        while True:
            event, values = self.window.read()
            try:
                handler = self.__getattribute__(f'handle_{event}')
                ret = handler(values)
                if ret is not None:
                    break
            except AttributeError:
                pass
            (ww, wh) = self.window.Size
            print(ww, wh)

            img = self.images[self.page]
            (h, w, *_) = img.shape
            w_prop = ((ww / 2) - 20) / w
            h_prop = (wh - 50) / h
            norm_scale = self.scale * min(w_prop, h_prop)
            print(self.scale, norm_scale)
            w = int(w*norm_scale)
            h = int(h*norm_scale)

            transformed_img = self.frame_transform(img)
            img = cv2.resize(img, fx = norm_scale, fy = norm_scale, dsize = (w, h))
            transformed_img = cv2.resize(transformed_img, fx = norm_scale, fy = norm_scale, dsize = (w, h))
            self.window['img'].update(array_to_data(img))
            self.window['img'].set_size((w, h))
            self.window['img2'].update(array_to_data(transformed_img))
            self.window['img2'].set_size((w, h))
            self.window['label'].update(value = f'Page {self.page + 1} de {len(self.images)}')
        self.window.close()
