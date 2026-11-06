"""Render the wall BEV the classifier sees, with its passage prediction.

Drawn in the same frame as vnm_ros's CARE BEV: forward is up, the robot's
left is the left of the image and the axes are in metres.
"""
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from corridor_classifier.passage_directions import (
    CLASS_TO_DIRECTIONS,
    DIRECTION_NAMES,
)


DISPLAY_ORDER = ("left", "front", "right")
OPEN_COLOR = (35, 150, 70)
CLOSED_COLOR = (175, 175, 175)
CLOSED_FILL = (95, 95, 95)
STALE_COLOR = (200, 40, 40)
TEXT_COLOR = (30, 30, 30)
GRID_COLOR = (225, 225, 225)
WALL_COLOR = np.array((35, 75, 170), dtype=np.float32)
ROBOT_COLOR = (225, 90, 30)


def _font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _ticks(minimum: float, maximum: float, spacing: float):
    first = np.ceil(minimum / spacing)
    last = np.floor(maximum / spacing)
    return [float(i) * spacing for i in np.arange(first, last + 1)]


def render_passage_bev(
    grid: Optional[np.ndarray],
    forward_range_m: Sequence[float],
    map_width_m: float,
    class_name: Optional[str],
    stale: bool = False,
    cell_pixels: int = 9,
    tick_spacing_m: float = 1.0,
    min_points: float = 0.0,
    binary: bool = False,
) -> Image.Image:
    """Return the overlay for one grid.

    ``grid`` is the (forward_bins, lateral_bins, channels) array from
    ``bev_occupancy_grid``, or None before the first cloud. Which directions
    are open follows from ``class_name``; a name with no directions, such as
    ``turning``, shows them as unknown. ``min_points`` and ``binary`` match
    the classifier's ``bev_min_points`` and ``bev_binary`` so the walls drawn
    are the ones the model receives.
    """
    forward_bins, lateral_bins = (
        (50, 70) if grid is None else (grid.shape[0], grid.shape[1])
    )
    width = lateral_bins * cell_pixels
    height = forward_bins * cell_pixels
    margin_left, margin_right, margin_bottom = 44, 14, 34
    header = 92
    footer = 96
    canvas = Image.new(
        "RGB",
        (
            margin_left + width + margin_right,
            header + height + margin_bottom + footer,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    origin = (margin_left, header)

    if grid is not None:
        counts = grid[..., 0].astype(np.float32)
        if min_points > 0.0:
            counts = counts * (counts >= min_points)
        if binary:
            strength = (counts > 0.0).astype(np.float32)
        else:
            counts = np.log1p(counts)
            peak = float(counts.max())
            strength = counts / peak if peak > 0.0 else counts
        # Row 0 is nearest the robot and column 0 is the far right, so flip
        # both to put forward at the top and the robot's left on the left.
        strength = strength[::-1, ::-1, None]
        pixels = 255.0 - strength * (255.0 - WALL_COLOR)
        cells = Image.fromarray(pixels.clip(0, 255).astype(np.uint8), "RGB")
        canvas.paste(
            cells.resize((width, height), Image.NEAREST),
            origin,
        )

    minimum_forward, maximum_forward = float(forward_range_m[0]), float(
        forward_range_m[1]
    )
    half_width = float(map_width_m) / 2.0

    def to_pixel(forward_m: float, left_m: float) -> Tuple[float, float]:
        x = origin[0] + (half_width - left_m) / (2.0 * half_width) * width
        y = (
            origin[1]
            + (maximum_forward - forward_m)
            / (maximum_forward - minimum_forward)
            * height
        )
        return x, y

    tick_font = _font(13)
    for tick in _ticks(-half_width, half_width, tick_spacing_m):
        x, _ = to_pixel(minimum_forward, tick)
        draw.line(
            [(x, origin[1]), (x, origin[1] + height)],
            fill=GRID_COLOR,
            width=1,
        )
        draw.text(
            (x - 6, origin[1] + height + 6),
            f"{abs(tick) if tick == 0 else -tick:g}",
            fill=TEXT_COLOR,
            font=tick_font,
        )
    for tick in _ticks(minimum_forward, maximum_forward, tick_spacing_m):
        _, y = to_pixel(tick, 0.0)
        draw.line(
            [(origin[0], y), (origin[0] + width, y)],
            fill=GRID_COLOR,
            width=1,
        )
        draw.text((12, y - 8), f"{tick:g}", fill=TEXT_COLOR, font=tick_font)
    draw.rectangle(
        [origin, (origin[0] + width, origin[1] + height)],
        outline=TEXT_COLOR,
        width=1,
    )

    robot_x, robot_y = to_pixel(minimum_forward, 0.0)
    draw.polygon(
        [
            (robot_x, robot_y - 18),
            (robot_x - 12, robot_y),
            (robot_x + 12, robot_y),
        ],
        fill=ROBOT_COLOR,
    )

    directions = CLASS_TO_DIRECTIONS.get(class_name)
    if directions is not None:
        ends = {
            "front": to_pixel(minimum_forward + 2.2, 0.0),
            "left": to_pixel(minimum_forward + 1.0, 1.8),
            "right": to_pixel(minimum_forward + 1.0, -1.8),
        }
        start = (robot_x, robot_y - 24)
        for name, is_open in zip(DIRECTION_NAMES, directions):
            if is_open:
                end = ends[name]
                draw.line([start, end], fill=OPEN_COLOR, width=7)
                radius = 9
                draw.ellipse(
                    [
                        end[0] - radius,
                        end[1] - radius,
                        end[0] + radius,
                        end[1] + radius,
                    ],
                    fill=OPEN_COLOR,
                )

    draw.text(
        (margin_left, 14),
        class_name or "waiting for /passage_type",
        fill=STALE_COLOR if stale else TEXT_COLOR,
        font=_font(40),
    )
    draw.text(
        (margin_left, 64),
        "STALE" if stale else "passage class",
        fill=STALE_COLOR if stale else (120, 120, 120),
        font=_font(15),
    )

    top = header + height + margin_bottom + 8
    label_font = _font(18)
    state_font = _font(22)
    cell_width = width // 3
    for column, name in enumerate(DISPLAY_ORDER):
        left = margin_left + column * cell_width
        box = [(left + 4, top), (left + cell_width - 4, top + 72)]
        if directions is None:
            color, state = CLOSED_COLOR, "--"
        elif directions[DIRECTION_NAMES.index(name)]:
            color, state = OPEN_COLOR, "OPEN"
        else:
            color, state = CLOSED_FILL, "CLOSED"
        draw.rectangle(box, fill=color)
        draw.text((left + 16, top + 8), name, fill="white", font=label_font)
        draw.text((left + 16, top + 36), state, fill="white", font=state_font)
    return canvas
