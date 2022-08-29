from dataclasses import dataclass
from enum import Enum, unique

import numpy as np

from pdfnormalizer.utils import log


@dataclass(frozen=True)
class Element():
    x: float
    y: float
    sx: float
    sy: float
    depth: int


@unique
class SubdivisionAction(Enum):
    UNDEFINED = 0
    END_FIGURE = 1
    END_TEXT = 2
    END_THRASH = 3
    HORIZONTAL = 4
    VERTICAL = 5


@unique
class BoundingBoxHint(Enum):
    PAGE_CONTAINER        = 1          # todo o conteúdo da página sem espaço em branco
    SUP_CONTENT_BLOCK     = 2          # bloco que pega mais de um bloco de conteúdo, tipo as duas colunas de um artigo duas colunas
    CONTENT_BLOCK         = 3          # bloco de conteúdo, tipo o bloco de autores de um artigo ou uma coluna de conteúdo
    SUB_CONTENT_BLOCK     = 4          # um pedaço de um bloco de conteúdo mas que n chega a ser uma unidade de conteúdo
    PICTURE_BLOCK         = 5          # um bloco de imagem
    TEXT_BLOCK            = 6          # um bloco de texto
    SUB_TEXT_BLOCK        = 7          # chega a comer um pedaço do texto ou dividir em mais de um
    HEADER                = 8          # cabeçalho, geralmente remove
    FOOTER                = 9          # rodapé, geralmente remove
    END_OF_LINE           = 10         # elemento encontrado quando a recursão vai fundo demais


def all_line_is_color(line, color, threshold = 0.999):
    if line.shape[0] == 0:
        return True
    proportion = float(np.sum(line == color)) / line.shape[0]
    return proportion > threshold


def prepare_page_for_subdivision(img):
    import cv2
    (w, h, channels) = img.shape
    background_color = img[0, 0]
    mask = cv2.inRange(img, background_color, background_color)
    # kernel = np.zeros((3, 5), np.uint8)
    # kernel[:, 2] = 1
    # kernel[1, :] = 3
    # kernel[1, 2] = 10
    # mask = cv2.dilate(mask, kernel)
    # mask = 255 - mask
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    return mask


def trim_whitespace(img, sx=None, sy=None, x=0, y=0, bg_color=None, line_threshold = 1):
    if bg_color is None:
        bg_color = img[x, y]
    (w, h, *rest) = img.shape
    if sx is None or sy is None:
        sx = w
        sy = h
    for i in range(y, y + sy):  # cima pra baixo
        line = img[x:x+sx, i]
        if all_line_is_color(line, bg_color, threshold = line_threshold) and sy > 0:
            y += 1
            sy -= 1
            continue
        break
    for i in reversed(range(y - 1, y+sy)):  # baixo pra cima
        line = img[x:x+sx, i]
        if all_line_is_color(line, bg_color, threshold = line_threshold) and sy > 0:
            sy -= 1
            continue
        break
    for i in range(x, x+sx):  # esquerda pra direita
        line = img[i, y:y+sy]
        if all_line_is_color(line, bg_color, threshold = line_threshold) and sx > 0:
            x += 1
            sx -= 1
            continue
        break
    for i in reversed(range(x - 1, x+sx)):  # direita pra esquerda
        line = img[i, y:y+sy]
        if all_line_is_color(line, bg_color, threshold = line_threshold) and sx > 0:
            sx -= 1
            continue
        break
    return (x, y, sx, sy)


def get_bounding_boxes(
        img,                # imagem de entrada
        bg_color=None,      # cor de fundo, primeiro pixel do canto
        depth=1,            # niveis de recursão já feitos
        horizontal=False,   # processando horizontalmente?
        sx=None,            # tamanho da bounding box x
        sy=None,            # tamanho da bounding box x
        x=0,                # começo da bounding box
        y=0,                # começo da bounding box
        max_depth=20,       # profundidade máxima
        background_threshold=0.999
        ):
    ret = []
    (w, h, *rest) = img.shape
    if sx is None or sy is None:
        sx = w
        sy = h
    if sx == 0 or sy == 0:
        return []
    if bg_color is None:
        bg_color = img[x, y]
    # if all_line_is_color(np.reshape(img[x:x+sx, y:y+sy], (sx*sy, -1)), bg_color, threshold = background_threshold):
    #     return []
    if background_threshold <= 0.1:
        log("threshold too low")
        return [Element(x=x/w, y=y/h, sx=sx/w, sy=sy/h, depth=depth)]
    (x, y, sx, sy) = trim_whitespace(img, sx=sx, sy=sy, x=x, y=y, bg_color=bg_color, line_threshold = background_threshold)
    # log('depth', depth, max_depth, 'unwhitespaced', x, y, sx, sy, 'threshold', background_threshold)
    if depth < max_depth:
        biggest_gap = 0
        biggest_gap_idx = 0
        current_gap = 0
        if not horizontal:
            for i in range(y, y + sy):
                line = img[x:x+sx, i]
                if all_line_is_color(line, bg_color, threshold = background_threshold):
                    current_gap += 1
                    if current_gap > biggest_gap:
                        biggest_gap = current_gap
                        biggest_gap_idx = i
                else:
                    current_gap = 0
        else:
            for i in range(x, x + sx):
                line = img[i, y:y+sy]
                if all_line_is_color(line, bg_color, threshold = background_threshold):
                    current_gap += 1
                    if current_gap > biggest_gap:
                        biggest_gap = current_gap
                        biggest_gap_idx = i
                else:
                    current_gap = 0
        sections = []
        if (biggest_gap_idx - biggest_gap) <= 4:
            return get_bounding_boxes(
                img=img,
                bg_color=bg_color,
                depth=depth,
                horizontal=horizontal,
                sx=sx,
                sy=sy,
                x=x,
                y=y,
                max_depth=max_depth,
                background_threshold=background_threshold - 0.01
            )
        if not horizontal:
            sections.append(y)
            sections.append(biggest_gap_idx)
            sections.append(y + sy)
        else:
            sections.append(x)
            sections.append(biggest_gap_idx)
            sections.append(x + sx)
        # print("sections", sections)
        ret = []
        for i in range(len(sections) - 1):
            a = sections[i]
            b = sections[i + 1]
            if not horizontal:
                ret.append(Element(x=x/w, y=a/h, sx=sx/w, sy=(b-a)/h, depth=depth))
            else:
                ret.append(Element(x=a/w, y=y/h, sx=(b-a)/w, sy=sy/h, depth=depth))
        return ret
    return [Element(x=x/w, y=y/h, sx=sx/w, sy=sy/h, depth=depth)]
