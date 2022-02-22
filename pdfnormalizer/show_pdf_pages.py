from argparse import ArgumentParser
from pdf2image import convert_from_path
from pathlib import Path
import PySimpleGUI as sg
import numpy as np
import cv2
from screeninfo import get_monitors, Enumerator
from pdfnormalizer.utils import array_to_data

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', type = Path, help = "PDF file to process", required = True)
    parser.add_argument('-z', type = float, help = "zoom scale", default = 1)
    parser.add_argument('-n', type = int, help = "start at page", default = 1)
    args = parser.parse_args()

    scale = args.z
    page = args.n - 1

    images = convert_from_path(args.i)
    images_arr = list(map(np.array, images))
    qt_images = len(images)

    if page < 0 or page >= qt_images:
        raise Exception(f"Invalid page: {page + 1}")

    layout = [
        [
            sg.Button("<"),
            sg.Button("+"),
            sg.Button("X"),
            sg.Button("-"),
            sg.Button(">"),
            sg.Text(key = 'label')
        ],
        [
            sg.Image(key = 'img'),
            sg.Image(key = 'img2'),
        ]
    ]


    window = sg.Window("Imagens", layout, resizable = True, finalize = True, element_justification = 'c')
    window.bind('<Configure>', 'Configure')
    window.write_event_value('init', None)
    while True:
        event, values = window.read()
        if event == 'X' or event == sg.WIN_CLOSED:
            break
        if event == '+':
            scale += 0.1
        if event == '-':
            scale -= 0.1
        if event == '<':
            if page - 1 >= 0:
                page -= 1
        if event == '>':
            if page + 1 < qt_images:
                page += 1
        (ww, wh) = window.Size
        print(ww, wh)

        img = images_arr[page]
        (h, w, *_) = img.shape
        w_prop = ((ww / 2) - 20) / w
        h_prop = (wh - 50) / h
        norm_scale = scale * min(w_prop, h_prop)
        print(scale, norm_scale)
        w = int(w*norm_scale)
        h = int(h*norm_scale)

        img = cv2.resize(img, fx = norm_scale, fy = norm_scale, dsize = (w, h))
        window['img'].update(array_to_data(img))
        window['img'].set_size((w, h))
        window['img2'].update(array_to_data(img))
        window['img2'].set_size((w, h))
        window['label'].update(value = f'Página {page + 1} de {qt_images}')
    window.close()
