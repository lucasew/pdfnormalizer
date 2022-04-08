from dataclasses import dataclass
from enum import Enum, unique
 
import numpy as np
import cv2

@dataclass(frozen = True)
class Element():
    x: float
    y: float
    sx: float
    sy: float
    depth: int

@unique
class SubdivisionAction(Enum):
    UNDEFINED = 0
    END = 1
    HORIZONTAL = 2
    VERTICAL = 3

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

def all_line_is_color(line, color):
    return np.alltrue(line == color)

def prepare_page_for_subdivision(img):
    (w, h, channels) = img.shape
    background_color = img[0,0]
    mask = cv2.inRange(img, background_color, background_color)
    mask = 255 - mask
    kernel = np.zeros((3, 5), np.uint8)
    kernel[:, 2] = 1
    kernel[1, :] = 3
    kernel[1, 2] = 10
    mask = cv2.dilate(mask, kernel)
    mask = 255 - mask
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    return mask

def get_bounding_boxes(
        img,                # imagem de entrada
        bg_color = None,    # cor de fundo, primeiro pixel do canto
        depth = 1,          # niveis de recursão já feitos
        horizontal = False, # processando horizontalmente?
        sx = None,          # tamanho da bounding box x
        sy = None,          # tamanho da bounding box x
        tx = 0.013,         # tamanho minimo de gap entre 2 elementos
        ty = 0.006,         # tamanho minimo de gap entre 2 elementos
        x = 0,              # começo da bounding box
        y = 0,              # começo da bounding box
        max_depth = 20      # profundidade máxima
    ):
    (w, h, *rest) = img.shape
    if sx is None or sy is None:
        sx = w
        sy = h
    min_gap_x = int(w*tx)
    min_gap_y = int(h*ty)
    if sx == 0 or sy == 0:
        return []
    if bg_color is None:
        bg_color = img[x, y]
    if all_line_is_color(np.reshape(img[x:x+sx, y:y+sy], (sx*sy)), bg_color):
        return []
    for i in range(y, y + sy): # cima pra baixo
        line = img[x:x+sx, i]
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
    current_element = Element(
        x = x / w,
        y = y / h,
        sx = sx / w,
        sy = sy / h,
        depth = depth
    )
    yield current_element
    # print('depth', depth, max_depth)
    if depth < max_depth:
        biggest_gap = 0
        current_gap = 0
        if not horizontal:
            for i in range(y, y + sy):
                line = img[x:x+sx, i]
                if all_line_is_color(line, bg_color):
                    # print('gap')
                    current_gap += 1
                    if current_gap > biggest_gap:
                        biggest_gap = current_gap
                else:
                    current_gap = 0
        else:
            for i in range(x, x + sx):
                line = img[i, y:y+sy]
                if all_line_is_color(line, bg_color):
                    current_gap += 1
                    if current_gap > biggest_gap:
                        biggest_gap = current_gap
                else:
                    current_gap = 0
        biggest_gap = int(3*(biggest_gap / 4))
        if not horizontal:
            if biggest_gap < min_gap_x:
                yield from get_bounding_boxes(
                    img,
                    depth = depth + 1,
                    max_depth = max_depth,
                    bg_color = bg_color,
                    x = x,
                    y = y,
                    sx = sx,
                    sy = sy,
                    horizontal = not horizontal
                )
        else:
            if biggest_gap < min_gap_y:
                yield from get_bounding_boxes(
                    img,
                    depth = depth + 1,
                    max_depth = max_depth,
                    bg_color = bg_color,
                    x = x,
                    y = y,
                    sx = sx,
                    sy = sy,
                    horizontal = not horizontal
                )
        if biggest_gap <= 2:
            return
        sections = []
        current_gap = 0
        if not horizontal:
            sections.append(y)
            for i in range(y, y + sy):
                line = img[x:x+sx, i]
                if all_line_is_color(line, bg_color):
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
                    current_gap += 1
                else:
                    if current_gap >= biggest_gap:
                        sections.append(i)
                        current_gap = 0
            sections.append(x + sx)
        for i in range(len(sections) - 1):
            a = sections[i]
            b = sections[i + 1]
            if not horizontal:
                yield from get_bounding_boxes(
                    img,
                    depth = depth + 1,
                    max_depth = max_depth,
                    bg_color = bg_color,
                    x = x,
                    y = a,
                    sx = sx,
                    sy = b - a,
                    horizontal = not horizontal
                )
            else:
                yield from get_bounding_boxes(
                    img,
                    depth = depth + 1,
                    max_depth = max_depth,
                    bg_color = bg_color,
                    x = a,
                    y = y,
                    sx = b - a,
                    sy = sy,
                    horizontal = not horizontal
                )
