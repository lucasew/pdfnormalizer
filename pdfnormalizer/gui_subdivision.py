from argparse import ArgumentParser
from pdf2image import convert_from_path
from pathlib import Path
import PySimpleGUI as sg
import numpy as np
import cv2
from screeninfo import get_monitors, Enumerator
from pdfnormalizer.utils import array_to_data, GUI, GUIHandler

def all_line_is_color(line, color):
    return np.alltrue(line == color)
class CustomGUIHandler(GUIHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tx = 0.013
        self.ty = 0.006
        self.show_subdivision = True
        self.black_emptyspace = True
        self.max_depth = 1
    def subdivide(self, img, depthmap = None, depth = 1, bg_color = None, x = 0, y = 0, sx = None, sy = None, horizontal = False):
        (w, h, *rest) = img.shape
        if sx is None or sy is None:
            sx = w
            sy = h
        min_gap_x = int(w * self.tx)
        min_gap_y = int(h * self.ty)
        if sx == 0 or sy == 0:
            return depthmap
        if depthmap is None:
            depthmap = np.zeros((w, h), dtype = np.uint8)
        if bg_color is None:
            bg_color = img[x, y]
        if all_line_is_color(np.reshape(img[x:x+sx, y:y+sy], (sx*sy)), bg_color):
            return depthmap
        for i in range(y, y + sy): # cima pra baixo
            line = img[x:x+sx, i]
            if all_line_is_color(line, bg_color) and sy > 0:
                if self.black_emptyspace:
                    depthmap[x:x+sx, i] = 0
                y += 1
                sy -= 1
                continue
            break
        for i in reversed(range(y - 1, y+sy)): # baixo pra cima
            line = img[x:x+sx, i]
            if all_line_is_color(line, bg_color) and sy > 0:
                if self.black_emptyspace:
                    depthmap[x:x+sx, i] = 0
                sy -= 1
                continue
            break
        for i in range(x, x+sx): # esquerda pra direita
            line = img[i, y:y+sy]
            if all_line_is_color(line, bg_color) and sx > 0:
                if self.black_emptyspace:
                    depthmap[i, y:y+sy] = 0
                x += 1
                sx -= 1
                continue
            break
        for i in reversed(range(x - 1, x+sx)): # direita pra esquerda
            line = img[i, y:y+sy]
            if all_line_is_color(line, bg_color) and sx > 0:
                if self.black_emptyspace:
                    depthmap[i, y:y+sy] = 0
                sx -= 1
                continue
            break
        # print("nivel", depth, x, sx, y, sy)
        depthmap[x:x+sx, y:y+sy] = int((depth * 255) / self.max_depth)
        if depth > self.max_depth:
            return depthmap
        if not horizontal:
            if sx < min_gap_x:
                return depthmap
            if sy < min_gap_y:
                return depthmap
        if sx > 2 and sy > 2:
            biggest_gap = 0
            current_gap = 0
            if not horizontal:
                for i in range(y, y + sy):
                    line = img[x:x+sx, i]
                    if all_line_is_color(line, bg_color):
                        # print('gap')
                        if self.black_emptyspace:
                            depthmap[x:x+sx, i] = 0
                        current_gap += 1
                        if current_gap > biggest_gap:
                            biggest_gap = current_gap
                    else:
                        current_gap = 0
            else:
                for i in range(x, x + sx):
                    line = img[i, y:y+sy]
                    if all_line_is_color(line, bg_color):
                        # print('gap')
                        if self.black_emptyspace:
                            depthmap[i, y:y+sy] = 0
                        current_gap += 1
                        if current_gap > biggest_gap:
                            biggest_gap = current_gap
                    else:
                        current_gap = 0
            # print('biggest_gap', biggest_gap, 'min', min_gap_x, min_gap_y)
            biggest_gap = int(3*(biggest_gap / 4))
            if not horizontal:
                if biggest_gap < min_gap_x:
                    return self.subdivide(
                        img,
                        depthmap,
                        depth = depth + 1,
                        bg_color = bg_color,
                        x = x,
                        y = y,
                        sx = sx,
                        sy = sy,
                        horizontal = not horizontal
                    )
            else:
                if biggest_gap < min_gap_y:
                    return self.subdivide(
                        img,
                        depthmap,
                        depth = depth + 1,
                        bg_color = bg_color,
                        x = x,
                        y = y,
                        sx = sx,
                        sy = sy,
                        horizontal = not horizontal
                    )
            if biggest_gap <= 2:
                return depthmap
            sections = []
            current_gap = 0
            if not horizontal:
                sections.append(y)
                for i in range(y, y + sy):
                    line = img[x:x+sx, i]
                    if all_line_is_color(line, bg_color):
                        if self.black_emptyspace:
                            depthmap[x:x+sx, i] = 0
                        current_gap += 1
                    else:
                        if current_gap >= biggest_gap:
                            sections.append(i)
                            current_gap = 0
                sections.append(y + sy)
            else:
                sections.append(x)
                for i in range(x, x + sx):
                    line = img[i, y:y+sy]
                    if all_line_is_color(line, bg_color):
                        if self.black_emptyspace:
                            depthmap[i, y:y+sy] = 0
                        current_gap += 1
                    else:
                        if current_gap >= biggest_gap:
                            sections.append(i)
                            current_gap = 0
                sections.append(x + sx)
            # print('sections', sections)
            if len(sections) >= 2:
                dmap = depthmap
                for i in range(len(sections) - 1):
                    a = sections[i]
                    b = sections[i + 1]
                    if not horizontal:
                        dmap = self.subdivide(
                            img,
                            depthmap = dmap,
                            depth = depth + 1,
                            bg_color = bg_color,
                            x = x,
                            y = a,
                            sx = sx,
                            sy = b - a,
                            horizontal = not horizontal
                        )
                    else:
                        dmap = self.subdivide(
                            img,
                            depthmap = dmap,
                            depth = depth + 1,
                            bg_color = bg_color,
                            x = a,
                            y = y,
                            sx = b - a,
                            sy = sy,
                            horizontal = not horizontal
                        )
                return dmap
            else:
                return self.subdivide(
                    img,
                    depthmap = dmap,
                    depth = depth + 1,
                    bg_color = bg_color,
                    x = x,
                    y = y,
                    sx = sx,
                    sy = sy,
                    horizontal = not horizontal
                )
        return depthmap

    def frame_transform(self, img):
        (w, h, channels) = img.shape
        background_color = img[0, 0]
        mask = cv2.inRange(img, background_color, background_color)
        mask = 255 - mask
        kernel = np.zeros((3, 5), np.uint8)
        kernel[:, 2] = 1
        kernel[1, :] = 3
        kernel[1, 2] = 10
        print(kernel)
        mask = cv2.dilate(mask, kernel)
        mask = 255 - mask
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        if not self.show_subdivision:
            return mask
        else:
            subdivided = self.subdivide(mask)
            max_value = np.max(subdivided)
            print(max_value)
            subdivided = np.array(subdivided * (255 / max_value), np.uint8)
            print(np.max(subdivided))
            return subdivided

    def handle_tx_up(self, gui, value):
        if self.tx < 1:
            self.tx += 0.01
            gui.emit('tick')
    def handle_tx_down(self, gui, value):
        if self.tx > 0:
            self.tx -= 0.001
            gui.emit('tick')
    def handle_ty_up(self, gui, value):
        if self.ty <= 1:
            self.ty += 0.01
            gui.emit('tick')
    def handle_ty_down(self, gui, value):
        if self.ty > 0:
            self.ty -= 0.001
            gui.emit('tick')
    def handle_max_depth_up(self, gui, value):
        if self.max_depth < 50:
            self.max_depth += 1
            gui.emit('tick')
    def handle_max_depth_down(self, gui, value):
        if self.max_depth > 1:
            self.max_depth -= 1
            gui.emit('tick')

    def handle_tick(self, gui, value):
        super().handle_tick(gui, value)
        gui.window['status'].update(value = f'tx: {self.tx:.4f} ty: {self.ty:.4f} d: {self.max_depth}')
    def handle_toggle_view(self, gui, value):
        self.show_subdivision = not self.show_subdivision
        gui.emit('tick')
    def handle_toggle_black_emptyspace(self, gui, value):
        self.black_emptyspace = not self.black_emptyspace
        gui.emit('tick')

    @property
    def buttons(self):
        return [
            *super().buttons,
            sg.Button('tx-', key = 'tx_down'),
            sg.Button('tx+', key = 'tx_up'),
            sg.Button('ty-', key = 'ty_down'),
            sg.Button('ty+', key = 'ty_up'),
            sg.Button('tg', key = 'toggle_view'),
            sg.Button('tbe', key = 'toggle_black_emptyspace'),
            sg.Button('d+', key = 'max_depth_up'),
            sg.Button('d-', key = 'max_depth_down'),
            sg.Text(key = 'status')
        ]

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', type = Path, help = "PDF file to process", required = True)
    parser.add_argument('-z', type = float, help = "zoom scale", default = 1)
    parser.add_argument('-n', type = int, help = "start at page", default = 1)
    args = parser.parse_args()

    scale = args.z
    page = args.n - 1

    images = convert_from_path(args.i)
    qt_images = len(images)

    if page < 0 or page >= qt_images:
        raise Exception(f"Invalid page: {page + 1}")

    handler = CustomGUIHandler(page = page, scale = scale)
    gui = GUI(images, handler)
    gui.run()
