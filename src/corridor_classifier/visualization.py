import cv2
import numpy as np
from sensor_msgs.msg import Image


def make_label_image_message(
    text: str,
    bridge,
    width: int = 480,
    height: int = 120,
) -> Image:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    origin = (
        max((width - text_width) // 2, 0),
        (height + text_height) // 2,
    )
    cv2.putText(
        canvas,
        text,
        origin,
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
