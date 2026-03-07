from dataclasses import dataclass
from enum import Enum, unique

import numpy as np

from pdfnormalizer.utils import log


@dataclass(frozen=True)
class Element():
    """
    Represents a recognized rectangular region (bounding box) in the image.

    Coordinates and dimensions are typically normalized (0.0 to 1.0) relative to the
    parent image size to remain resolution-independent.

    Attributes:
        x: The horizontal starting coordinate (top-left).
        y: The vertical starting coordinate (top-left).
        sx: The width of the region (size x).
        sy: The height of the region (size y).
        depth: The recursion depth at which this element was discovered, useful for
               tracking nested structures or forcing an early stop when parsing deeply.
    """
    x: float
    y: float
    sx: float
    sy: float
    depth: int


@unique
class SubdivisionAction(Enum):
    """
    Defines the layout classification or structural action to take on a bounding box.

    Used by the predictive model to classify a region as an atomic unit (text/figure/trash)
    or to signal that the region needs further splitting either horizontally or vertically.
    """
    UNDEFINED = 0
    END_FIGURE = 1
    END_TEXT = 2
    END_THRASH = 3
    HORIZONTAL = 4
    VERTICAL = 5


@unique
class BoundingBoxHint(Enum):
    """
    Semantic hints providing contextual classification for recognized blocks.

    These hints categorize areas based on their expected document structure
    (e.g., distinguishing between a full page wrapper and a specific text block).
    """
    PAGE_CONTAINER        = 1          # Full page content without surrounding whitespace
    SUP_CONTENT_BLOCK     = 2          # A wrapper encompassing multiple logical blocks (e.g., a multi-column layout area)
    CONTENT_BLOCK         = 3          # A distinct logical block, like an author list or a single text column
    SUB_CONTENT_BLOCK     = 4          # A fragment of a content block that hasn't formed a full logical unit
    PICTURE_BLOCK         = 5          # A figure or image region
    TEXT_BLOCK            = 6          # A self-contained text area
    SUB_TEXT_BLOCK        = 7          # A text fragment that potentially splits a text block
    HEADER                = 8          # Page header, usually discarded during extraction
    FOOTER                = 9          # Page footer, usually discarded during extraction
    END_OF_LINE           = 10         # The terminal node when recursive subdivision hits the maximum depth


def all_line_is_color(line, color, threshold = 0.999):
    """
    Evaluates if an image line (row or column) is overwhelmingly composed of a target color.

    This acts as a heuristic to detect structural gaps (like whitespace or separators) by
    calculating the ratio of pixels matching the provided background color.

    Args:
        line: A 1D numpy array representing a slice of the image.
        color: The target background pixel value.
        threshold: The required ratio (0.0 to 1.0) of target color to be considered a clear line.
                   A lower threshold makes the check more tolerant to noise (e.g., scanning artifacts).

    Returns:
        True if the proportion of 'color' pixels strictly exceeds 'threshold', False otherwise.
    """
    if line.shape[0] == 0:
        return True
    proportion = float(np.sum(line == color)) / line.shape[0]
    return proportion > threshold


def prepare_page_for_subdivision(img):
    """
    Transforms a raw document image into a high-contrast binary mask to simplify spatial analysis.

    Assumes the very first pixel (top-left, [0, 0]) represents the overall document background.
    It isolates this background using cv2.inRange and applies a strict threshold to binarize
    the result. This uniform mask prevents variations in paper color or slight gradients
    from disrupting subsequent whitespace-trimming or gap-detection logic.

    Args:
        img: A 3D numpy array representing the input image (width, height, channels).

    Returns:
        A 2D binary numpy array mask where content and background are rigidly separated.
    """
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
    """
    Reduces the bounds of a given region by removing empty background margins from all four sides.

    This function iterativelly shrinks the active bounding box (defined by x, y, sx, sy) inward
    until it hits a structural boundary (a line that drops below the 'line_threshold' for bg_color).
    It ensures that extraction or subsequent subdivision always operates tightly on actual content,
    improving precision and discarding useless padding.

    Args:
        img: The image mask array to be processed.
        sx, sy: Current width and height to process. Defaults to full image size.
        x, y: Starting top-left coordinate. Defaults to (0, 0).
        bg_color: The color treated as empty space. Derived from img[x, y] if omitted.
        line_threshold: Tolerance for treating a line as empty (passed to all_line_is_color).

    Returns:
        A tuple (x, y, sx, sy) representing the tightest bounding box encompassing the content.
    """
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
    """
    Recursively discovers sub-regions within an image block by scanning for the largest background gaps.

    The algorithm performs a single subdivision pass on the given axis (horizontal or vertical).
    It first tightly crops the region, then scans perpendicularly to find the widest contiguous
    block of background color. If a significant gap is found, the region splits into two or more
    sub-elements.

    To handle noisy scans or subtle overlaps, the function dynamically lowers the `background_threshold`
    and retries if the largest gap found is unexpectedly thin (<= 4 pixels wide). The recursion
    deepens until 'max_depth' is reached or no further subdivisions can be established.

    Args:
        img: Mask array indicating content and layout.
        bg_color: Target background pixel value; inferred dynamically if not provided.
        depth: The current recursion index.
        horizontal: Determines the scanning axis. If True, it splits the layout horizontally
                    (scanning top-to-bottom or left-to-right depending on orientation logic).
        sx, sy: Current bounding dimensions for the region.
        x, y: Top-left coordinate anchoring the current region.
        max_depth: A safety limit to prevent infinite recursion on highly granular images.
        background_threshold: The purity ratio required to classify a scanned line as a gap.

    Returns:
        A list of `Element` objects, mapping normalized normalized coordinates and sizes.
    """
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
