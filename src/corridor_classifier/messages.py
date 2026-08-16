from typing import Sequence

from scenario_navigation_msgs.msg import cmd_dir_intersection


def make_passage_message(
    class_index: int,
    class_names: Sequence[str],
) -> cmd_dir_intersection:
    if class_index < 0 or class_index >= len(class_names):
        raise ValueError(
            f"class_index must be in [0, {len(class_names) - 1}]: {class_index}"
        )

    msg = cmd_dir_intersection()
    msg.cmd_dir = [0, 0, 0]
    one_hot = [0] * len(msg.intersection_label)
    if class_index < len(one_hot):
        one_hot[class_index] = 1
    msg.intersection_label = one_hot
    msg.intersection_name = str(class_names[class_index])
    return msg
