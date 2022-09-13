import numpy as np

def log(*args, **kwargs):
    from sys import stderr
    print(file=stderr, *args, **kwargs)

def array_to_data(array):
    from PIL import Image
    from io import BytesIO
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data

class Exporter:
    @staticmethod
    def pre():
        return '''
<!DOCTYPE html>
<html>
<head>
    <style>
    body {
        max-width: 100vw;
    }
    body > * {
        max-width: 100%;
    }
    </style>
    <meta charset="UTF-8"/>
</head>
<body>
'''
    @staticmethod
    def pos():
        return "</body></html>"
    @staticmethod
    def block(img, kind, x, sx, y, sy, depth):
        from pytesseract import image_to_string
        ret = ""
        roi = img[x:x+sx, y:y+sy]
        if kind == "text":
            ret += "<!-- Início do bloco de texto -->"
            for line in image_to_string(roi).split("\n"):
                if line != "":
                    ret += f"<p>{line}</p>"
        elif kind == "figure":
            b64 = b64encode(array_to_data(roi))
            ret += f"<img src=data:image/png;base64,{b64.decode('ascii')}>"
        return ret

class GUIHandler():
    def __init__(self, scale=1.0, page=0, **kwargs):
        self.scale = scale
        self.page = page

    def frame_transform(self, image):
        return image, image

    def handle_image_change(self, gui):
        import cv2
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
        img = cv2.resize(img, fx=norm_scale, fy=norm_scale, dsize=(w, h))
        transformed_img = cv2.resize(transformed_img, fx=norm_scale, fy=norm_scale, dsize=(w, h))
        gui.window['img'].update(array_to_data(img))
        gui.window['img'].set_size((w, h))
        gui.window['img2'].update(array_to_data(transformed_img))
        gui.window['img2'].set_size((w, h))
        gui.window['label'].update(value=f'Page {self.page + 1} de {len(gui.images)}')

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
        import PySimpleGUI as sg
        return [
            sg.Button("<", key='prev'),
            sg.Button("+", key='more_zoom'),
            sg.Button("X"),
            sg.Button("-", key='less_zoom'),
            sg.Button(">", key='next'),
            sg.Text(key='label')
        ]


class GUI():
    def layout(self):
        import PySimpleGUI as sg
        return [
            self.handler.buttons,
            [sg.Image(key='img'), sg.Image(key='img2')]
        ]

    def __init__(self, images=[], handler=GUIHandler()):
        import PySimpleGUI as sg
        self.handler = handler
        self.images = list(map(np.array, images))
        self.window = sg.Window('window', self.layout(), resizable=True, finalize=True, element_justification='c')
        self.window.bind('<Configure>', 'window_event')
        self.emit('tick')
    def emit(self, name='tick', value=None):
        self.window.write_event_value(name, value)
    def tick(self, timeout=None):
        event, values = self.window.read(timeout=timeout)
        handler = None
        try:
            handler = self.handler.__getattribute__(f'handle_{event}')
            ret = handler(self, values)
            if ret is not None:
                return False
        except AttributeError:
            pass
        log("GUI event: ", event, handler is not None, values)
        return True

    def run(self):
        while self.tick():
            pass
        self.window.close()

class sigmoid_focal_crossentropy_loss():
    def __init__(
        self,
        alpha = 0.25,
        gamma = 0.,
        from_logits: bool = False,
    ):
        import tensorflow as tf
        if gamma and gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero.")
        self.gamma = tf.constant(gamma)
        self.alpha = tf.constant(alpha)
        self.from_logits = from_logits

    def get_config(self):
        return dict(alpha=float(self.alpha), gamma=float(self.gamma), from_logits=self.from_logits)

    def __call__(self, y_true, y_pred):
        import tensorflow as tf
        from tensorflow.keras import backend as K
        # y_true = tf.one_hot(y_true, NUM_CLASSES)[0]
        # y_pred = tf.cast(y_pred)
        # y_true = tf.cast(y_true, dtype=y_pred.dtype)

        # Get the cross_entropy for each entry
        ce = K.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)

        # If logits are provided then convert the predictions into probabilities
        if self.from_logits:
            pred_prob = tf.sigmoid(y_pred)
        else:
            pred_prob = y_pred

        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        # alpha_factor = 1.0
        # modulating_factor = 1.0

        # if alpha:
        alpha = tf.cast(self.alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

        # if gamma:
        gamma = tf.cast(self.gamma, dtype=y_true.dtype)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

        # compute the final loss and return
        return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)


