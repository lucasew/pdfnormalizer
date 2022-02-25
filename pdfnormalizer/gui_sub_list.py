from argparse import ArgumentParser
from pdf2image import convert_from_path
from pathlib import Path
import PySimpleGUI as sg
import numpy as np
import cv2
from screeninfo import get_monitors, Enumerator
from pdfnormalizer.utils import array_to_data, GUI, GUIHandler
from pdfnormalizer.model import Element, prepare_page_for_subdivision, get_bounding_boxes

def all_line_is_color(line, color):
    return np.alltrue(line == color)
class CustomGUIHandler(GUIHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tx = 0.013
        self.ty = 0.006
        self.show_subdivision = True
        self.black_emptyspace = True
        self.only_last_level = False
        self.max_depth = 1
    def frame_transform(self, img):
        prepared = prepare_page_for_subdivision(img)
        print(get_bounding_boxes(prepared))
        return prepared
    # def handle_tx_up(self, gui, value):
    #     if self.tx < 1:
    #         self.tx += 0.01
    #         gui.emit('tick')
    # def handle_tx_down(self, gui, value):
    #     if self.tx > 0:
    #         self.tx -= 0.001
    #         gui.emit('tick')
    # def handle_ty_up(self, gui, value):
    #     if self.ty <= 1:
    #         self.ty += 0.01
    #         gui.emit('tick')
    # def handle_ty_down(self, gui, value):
    #     if self.ty > 0:
    #         self.ty -= 0.001
    #         gui.emit('tick')
    # def handle_max_depth_up(self, gui, value):
    #     if self.max_depth < 50:
    #         self.max_depth += 1
    #         gui.emit('tick')
    # def handle_max_depth_down(self, gui, value):
    #     if self.max_depth > 1:
    #         self.max_depth -= 1
    #         gui.emit('tick')
    # def handle_toggle_view(self, gui, value):
    #     self.show_subdivision = not self.show_subdivision
    #     gui.emit('tick')
    # def handle_toggle_only_last_level(self, gui, value):
    #     self.only_last_level = not self.only_last_level
    #     gui.emit('tick')
    # def handle_toggle_black_emptyspace(self, gui, value):
    #     self.black_emptyspace = not self.black_emptyspace
    #     gui.emit('tick')


    def handle_tick(self, gui, value):
        super().handle_tick(gui, value)
        gui.window['status'].update(value = f'tx: {self.tx:.4f} ty: {self.ty:.4f} d: {self.max_depth}')
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
            sg.Button('oll', key = 'toggle_only_last_level'),
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
