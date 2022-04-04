from argparse import ArgumentParser
from pdf2image import convert_from_path
from pathlib import Path
import PySimpleGUI as sg
import numpy as np
import cv2
from screeninfo import get_monitors, Enumerator
from pdfnormalizer.utils import array_to_data, GUI, GUIHandler

class CustomGUIHandler(GUIHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = kwargs.get('threshold') or 128
    def handle_threshold_change(self, gui):
        gui.window['threshold'].update(value = f'Threshold: {self.threshold}/255')
    def handle_tick(self, gui, value):
        super().handle_tick(gui, value)
        self.handle_threshold_change(gui)
    def handle_init(self, gui, value):
        super().handle_init()
    def handle_threshold_up(self, gui, value):
        if self.threshold < 255:
            self.threshold += 1
            gui.emit('tick')
    def handle_threshold_down(self, gui, value):
        if self.threshold > 0:
            self.threshold -= 1
            gui.emit('tick')
    def frame_transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, img = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)
        return img
    @property
    def buttons(self):
        return [
            *super().buttons,
            sg.Button('t-1', key = 'threshold_down'),
            sg.Button('t+1', key = 'threshold_up'),
            sg.Text(key = 'threshold')
        ]

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', type = Path, help = "PDF file to process", required = True)
    parser.add_argument('-z', type = float, help = "zoom scale", default = 1)
    parser.add_argument('-n', type = int, help = "start at page", default = 1)
    parser.add_argument('-t', type = int, help = "start at threshold", default = 128)
    args = parser.parse_args()

    scale = args.z
    page = args.n - 1
    threshold = args.t

    images = convert_from_path(args.i)
    qt_images = len(images)

    if page < 0 or page >= qt_images:
        raise Exception(f"Invalid page: {page + 1}")

    handler = CustomGUIHandler(page = page, scale = scale, threshold = threshold)
    gui = GUI(images, handler)
    gui.run()