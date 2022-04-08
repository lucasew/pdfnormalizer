from argparse import ArgumentParser
from pdf2image import convert_from_path
from pathlib import Path
import PySimpleGUI as sg
import numpy as np
import cv2
import random
from screeninfo import get_monitors, Enumerator
from pdfnormalizer.utils import array_to_data, GUI, GUIHandler
from pdfnormalizer.model import Element, prepare_page_for_subdivision, get_bounding_boxes, SubdivisionAction

class CustomGUIHandler(GUIHandler):
    current_element = None
    elemsh = []
    elemsv = []
    known_elements = {}

    def get_subdivision(self, elem):
        if self.known_elements.get(self.page) is None:
            self.known_elements[self.page] = {}
        k = (int(elem.x * 100), int(elem.y * 100), int(elem.sx * 100), int(elem.sy * 100))
        elem, sub = self.known_elements[self.page].get(k)
        if r is None:
            return SubdivisionAction.UNDEFINED
        return sub
    def set_subdivision(self, elem, sub, replace = False):
        if self.known_elements.get(self.page) is None:
            self.known_elements[self.page] = {}
        k = (int(elem.x * 100), int(elem.y * 100), int(elem.sx * 100), int(elem.sy * 100))
        if self.known_elements[self.page].get(k) is None or replace:
            self.known_elements[self.page][k] = (elem, sub)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def handle_page_change(self, gui):
        print('current_element', self.current_element)
        gui.emit('tick')
    def handle_prev(self, gui, value):
        super().handle_prev(gui, value)
        self.handle_page_change(gui)
    def handle_next(self, gui, value):
        super().handle_next(gui, value)
        self.handle_page_change(gui)
    def frame_transform(self, img):
        self.current_element = None
        self.set_subdivision(Element(x = 0, y = 0, sx = 1, sy = 1, depth = 1), SubdivisionAction.UNDEFINED)
        for (_, (elem, cls)) in self.known_elements[self.page].items():
            if cls is SubdivisionAction.UNDEFINED:
                self.current_element = elem
                break
        if self.current_element is None:
            return img
        imgh = img.copy()
        imgv = img.copy()
        w, h, *_ = img.shape
        # print(w, h, _)
        depth = self.current_element.depth
        prepared = prepare_page_for_subdivision(img)
        bbh = [*get_bounding_boxes(prepared,
            horizontal = True,
            depth = depth,
            max_depth = depth + 1,
            sx = int(self.current_element.sx * w),
            sy = int(self.current_element.sy * h),
            x = int(self.current_element.x * w),
            y = int(self.current_element.y * h)
        )]
        self.elemsh = bbh
        bbv = [*get_bounding_boxes(prepared,
            horizontal = False,
            depth = depth,
            max_depth = depth + 1,
            sx = int(self.current_element.sx * w),
            sy = int(self.current_element.sy * h),
            x = int(self.current_element.x * w),
            y = int(self.current_element.y * h)
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
        if self.current_element is not None:
            self.set_subdivision(self.current_element, SubdivisionAction.HORIZONTAL, replace = True)
            if len(self.elemsh) > 0:
                for elem in self.elemsh:
                    self.set_subdivision(elem, SubdivisionAction.UNDEFINED)
            self.current_element = None
            gui.emit('tick')
    def handle_selectv(self, gui, value):
        if self.current_element is not None:
            self.set_subdivision(self.current_element, SubdivisionAction.VERTICAL, replace = True)
            if len(self.elemsv) > 0:
                for elem in self.elemsv:
                    self.set_subdivision(elem, SubdivisionAction.UNDEFINED)
            self.current_element = None
            gui.emit('tick')

    def handle_eol(self, gui, value):
        if self.current_element is not None:
            self.set_subdivision(self.current_element, SubdivisionAction.END, replace = True)
            self.current_element = None
            gui.emit('tick')

    def handle_selectlixo(self, gui, value):
        if self.current_element is not None:
            self.set_subdivision(self.current_element, SubdivisionAction.THRASH, replace = True)
            self.current_element = None
            gui.emit('tick')

    def handle_tick(self, gui, value):
        self.handle_image_change(gui)
        super().handle_tick(gui, value)
        gui.window['status'].update(value = f'tx: {self.tx:.4f} ty: {self.ty:.4f} d: {self.max_depth}')

    @property
    def buttons(self):
        return [
            *super().buttons,
            sg.Button('lixo', key = 'selectlixo'),
            sg.Button('vert. (verde)', key = 'selectv'),
            sg.Button('horiz. (vermelho)', key = 'selecth'),
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
