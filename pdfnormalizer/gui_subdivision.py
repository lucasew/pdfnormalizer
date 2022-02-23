from argparse import ArgumentParser
from pdf2image import convert_from_path
from pathlib import Path
import PySimpleGUI as sg
import numpy as np
import cv2
from screeninfo import get_monitors, Enumerator
from pdfnormalizer.utils import array_to_data, GUI, GUIHandler

def all_line_is_color(line, color):
    (r, g, b) = color
    check_r = line[:, 0] == r
    check_g = line[:, 1] == g
    check_b = line[:, 2] == b
    arr = np.array([check_r, check_g, check_b])
    (sx, sy) = arr.shape
    if sy == 1:
        return False
    return np.alltrue(arr)
class CustomGUIHandler(GUIHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def subdivide(self, img, depthmap = None, depth = 0, bg_color = None, x = 0, y = 0, sx = None, sy = None):
        if depth > 20:
            return depthmap
        print('subdivide chegou')
        (w, h, *rest) = img.shape
        if sx is None or sy is None:
            sx = w
            sy = h
        if sx == 0 or sy == 0:
            return depthmap
        if depthmap is None:
            depthmap = np.zeros((w, h), dtype = np.uint8)
        if bg_color is None:
            bg_color = img[x, y]
        if np.alltrue(np.array([
                img[x:x+sx, y:y+sy, 0] == bg_color[0],
                img[x:x+sx, y:y+sy, 1] == bg_color[1],
                img[x:x+sx, y:y+sy, 2] == bg_color[2]
            ])):
                return depthmap
        print(bg_color) #geralmente branco
        for i in range(y, y + sy): # cima pra baixo
            line = img[x:x+sx, i]
            print(line)
            print(all_line_is_color(line, bg_color))
            if all_line_is_color(line, bg_color) and sy > 0:
                y += 1
                sy -= 1
                continue
            break
        for i in reversed(range(y - 1, y+sy)): # baixo pra cima
            line = img[x:x+sx, i]
            if all_line_is_color(line, bg_color) and sy > 0:
                sy -= 1
                continue
            break
        for i in range(x, x+sx): # esquerda pra direita
            line = img[i, y:y+sy]
            if all_line_is_color(line, bg_color) and sx > 0:
                x += 1
                sx -= 1
                continue
            break
        for i in reversed(range(x - 1, x+sx)): # direita pra esquerda
            line = img[i, y:y+sy]
            if all_line_is_color(line, bg_color) and sx > 0:
                sx -= 1
                continue
            break
        print("nivel", depth, x, sx, y, sy)
        depthmap[x:x+sx, y:y+sy] = depth
        if sx > 1 and sy > 1:
            return self.subdivide(img, depthmap, depth = depth + 1, bg_color = bg_color, x = x, y = y, sx = sx, sy = sy)
        else:
            return depthmap

    def frame_transform(self, img):
        (w, h, channels) = img.shape
        return self.subdivide(img)
    # @property
    # def buttons(self):
    #     return [
    #         *super().buttons
    #     ]

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
