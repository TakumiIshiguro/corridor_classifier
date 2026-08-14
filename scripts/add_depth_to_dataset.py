#!/usr/bin/env python3
import argparse
import bisect
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import cv2
import numpy as np
import rosbag
import rospy
import yaml
from cv_bridge import CvBridge
from PIL import Image as PILImage
from tqdm.auto import tqdm

from corridor_classifier.config import load_collection_config, package_root, resolve_path
from corridor_classifier.dataset import session_directories
from unidepth_ros.config import (
    load_config as load_unidepth_config,
    repository_root as unidepth_repository_root,
    resolve_repository_path,
)
from unidepth_ros.depth_estimator import UniDepthV2Small


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=None)
    if argv is None:
        argv = sys.argv
    return parser.parse_args(rospy.myargv(argv=argv)[1:])


def _load_manifest(session_dir):
    path = os.path.join(session_dir, "samples.csv")
    with open(path, "r", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "stamp" not in rows[0]:
        raise ValueError(f"manifest has no timestamped samples: {path}")
    return path, rows


def _write_manifest(path, rows):
    fieldnames = list(rows[0].keys())
    if "depth_filename" not in fieldnames:
        fieldnames.insert(1, "depth_filename")
    temporary = path + ".tmp"
    with open(temporary, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _session_metadata(session_dir):
    path = os.path.join(session_dir, "metadata.yaml")
    with open(path, "r") as stream:
        metadata = yaml.safe_load(stream) or {}
    bag_path = str(metadata.get("bag_path", "")).strip()
    image_topic = str(metadata.get("image_topic", "")).strip()
    input_size = metadata.get("input_size", [224, 224])
    if not bag_path or not image_topic:
        raise ValueError(
            f"session metadata must contain bag_path and image_topic: {path}"
        )
    return bag_path, image_topic, [int(input_size[0]), int(input_size[1])]


def add_depth_to_session(session_dir, estimator, tolerance_seconds):
    manifest_path, rows = _load_manifest(session_dir)
    bag_path, image_topic, input_size = _session_metadata(session_dir)
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"source bag was not found: {bag_path}")
    targets = sorted(
        (float(row["stamp"]), index) for index, row in enumerate(rows)
    )
    target_stamps = [item[0] for item in targets]
    unmatched = set(range(len(rows)))
    depth_dir = os.path.join(session_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    bridge = CvBridge()
    progress = tqdm(
        total=len(rows), desc=os.path.basename(session_dir), unit="frame"
    )
    with rosbag.Bag(bag_path, "r") as bag:
        for _, message, bag_time in bag.read_messages(topics=[image_topic]):
            stamp = message.header.stamp.to_sec()
            if stamp <= 0.0:
                stamp = bag_time.to_sec()
            position = bisect.bisect_left(target_stamps, stamp)
            match = None
            for candidate in (position - 1, position):
                if 0 <= candidate < len(targets):
                    target_stamp, row_index = targets[candidate]
                    if (
                        row_index in unmatched
                        and abs(target_stamp - stamp) <= tolerance_seconds
                    ):
                        match = row_index
                        break
            if match is None:
                continue
            rgb = bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
            prediction = estimator.predict(PILImage.fromarray(rgb))
            depth = cv2.resize(
                prediction.depth_meters,
                (input_size[1], input_size[0]),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float16)
            relative_path = os.path.join("depth", f"{match:06d}.npy")
            np.save(
                os.path.join(session_dir, relative_path),
                depth,
                allow_pickle=False,
            )
            rows[match]["depth_filename"] = relative_path
            unmatched.remove(match)
            progress.update(1)
            if not unmatched:
                break
    progress.close()
    if unmatched:
        raise ValueError(
            f"could not match {len(unmatched)} dataset stamps in {bag_path}; "
            "increase depth_generation.stamp_tolerance_seconds only if needed"
        )
    _write_manifest(manifest_path, rows)


def main():
    args = parse_args()
    rospy.init_node("corridor_dataset_depth_generator")
    config = load_collection_config(args.config_dir)
    generation = config["depth_generation"]
    unidepth_config = load_unidepth_config(
        resolve_path(generation["unidepth_config_file"], package_root())
    )
    estimator = UniDepthV2Small(
        unidepth_config["depth"],
        unidepth_repository_root(),
        resolve_repository_path(unidepth_config["depth"]["model_path"]),
    )
    dataset_root = resolve_path(
        config["paths"]["dataset_dir"], package_root()
    )
    sessions = []
    for dataset_type in generation["dataset_types"]:
        sessions.extend(
            session_directories(os.path.join(dataset_root, dataset_type))
        )
    selected_names = set(generation["session_names"])
    if selected_names:
        sessions = [
            session
            for session in sessions
            if os.path.basename(session) in selected_names
        ]
    if not sessions:
        raise ValueError("no dataset sessions selected for depth generation")
    for session in sessions:
        add_depth_to_session(
            session,
            estimator,
            generation["stamp_tolerance_seconds"],
        )
    rospy.loginfo("added metric depth to %d dataset session(s)", len(sessions))


if __name__ == "__main__":
    main()
