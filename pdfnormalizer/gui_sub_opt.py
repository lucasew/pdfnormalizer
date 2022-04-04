from argparse import ArgumentParser
from pdf2image import convert_from_path
from pathlib import Path
import PySimpleGUI as sg
import numpy as np
import cv2
import random
from screeninfo import get_monitors, Enumerator
from pdfnormalizer.utils import array_to_data, GUI, GUIHandler
from pdfnormalizer.model import Element, prepare_page_for_subdivision, get_bounding_boxes

class CustomGUIHandler(GUIHandler):
    current_element = []
    elemsh = []
    elemsv = []
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def handle_page_change(self):
        self.current_element = []
        self.current_element.append(Element(x = 0, y = 0, sx = 1, sy = 1, depth = 1))
    def handle_prev(self, gui, value):
        super().handle_prev(gui, value)
        self.handle_page_change()
    def handle_next(self, gui, value):
        super().handle_next(gui, value)
        self.handle_page_change()
    def frame_transform(self, img):
        if len(self.current_element) < 1:
            self.handle_page_change()
        imgh = img.copy()
        imgv = img.copy()
        w, h, *_ = img.shape
        # print(w, h, _)
        depth = len(self.current_element)
        prepared = prepare_page_for_subdivision(img)
        bbh = [*get_bounding_boxes(prepared,
            horizontal = True,
            depth = depth,
            max_depth = depth + 1,
            sx = int(self.current_element[-1].sx * w),
            sy = int(self.current_element[-1].sy * h),
            x = int(self.current_element[-1].x * w),
            y = int(self.current_element[-1].y * h)
        )]
        self.elemsh = bbh
        bbv = [*get_bounding_boxes(prepared,
            horizontal = False,
            depth = depth,
            max_depth = depth + 1,
            sx = int(self.current_element[-1].sx * w),
            sy = int(self.current_element[-1].sy * h),
            x = int(self.current_element[-1].x * w),
            y = int(self.current_element[-1].y * h)
        )]
        self.elemsv = bbv
        for bb in bbh:
            # print('bbh', bb)
            x = int(bb.x * w)
            y = int(bb.y * h)
            sx = int((bb.x + bb.sx) * w)
            sy = int((bb.y + bb.sy) * h)
            imgh = cv2.rectangle(imgh, (y, x), (sy, sx), (255, 0, 0), 2)
        for bb in bbv:
            # print('bbv', bb)
            x = int(bb.x * w)
            y = int(bb.y * h)
            sx = int((bb.x + bb.sx) * w)
            sy = int((bb.y + bb.sy) * h)
            imgv = cv2.rectangle(imgv, (y, x), (sy, sx), (0, 255, 0), 2)
        return imgh, imgv
    def handle_selecth(self, gui, value):
        if len(self.elemsh) > 0:
            self.current_element.append(random.choice(self.elemsh))
            gui.emit('tick')
    def handle_selectv(self, gui, value):
        if len(self.elemsv) > 0:
            self.current_element.append(random.choice(self.elemsv))
            gui.emit('tick')
    def handle_eol(self, gui, value):
        if len(self.current_element) > 1:
            self.current_element.pop()
            gui.emit('tick')
    def handle_tick(self, gui, value):
        if len(self.current_element) == 0:
            self.handle_image_change(gui)
        super().handle_tick(gui, value)
        gui.window['status'].update(value = f'tx: {self.tx:.4f} ty: {self.ty:.4f} d: {self.max_depth}')
    @property
    def buttons(self):
        return [
            *super().buttons,
            sg.Button('horiz. (vermelho)', key = 'selecth'),
            sg.Button('vert. (verde)', key = 'selectv'),
            sg.Button('fim de linha', key = 'eol'),
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
