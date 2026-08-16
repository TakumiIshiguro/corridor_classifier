#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rosbag
import yaml

from corridor_classifier.config import (
    load_collection_config,
    package_root,
    resolve_path,
)
from corridor_classifier.dataset import session_directories
from corridor_classifier.turn_detection import (
    bridge_short_turning_gaps,
    detector_from_pose_messages,
    replace_post_turn_label_islands,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a lightweight copy of a corridor dataset and relabel "
            "MCL-detected turning frames."
        )
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-dir", default=None)
    return parser.parse_args()


def _load_detector(bag_path, config):
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"source bag was not found: {bag_path}")
    pose_topic = config["pose_topic"]
    with rosbag.Bag(bag_path, "r") as bag:
        available = bag.get_type_and_topic_info().topics
        if pose_topic not in available:
            raise ValueError(f"source bag does not contain {pose_topic}: {bag_path}")
        return detector_from_pose_messages(
            bag.read_messages(topics=[pose_topic]),
            angular_speed_threshold_rad_s=config[
                "angular_speed_threshold_rad_s"
            ],
            window_seconds=config["window_seconds"],
            minimum_duration_seconds=config["minimum_duration_seconds"],
            padding_seconds=config["padding_seconds"],
            maximum_pose_gap_seconds=config["maximum_pose_gap_seconds"],
        )


def _link_payload_directories(source_session, output_session):
    for name in ("images", "depth"):
        source = os.path.join(source_session, name)
        if not os.path.isdir(source):
            continue
        target = os.path.join(output_session, name)
        relative_source = os.path.relpath(source, output_session)
        os.symlink(relative_source, target, target_is_directory=True)


def _relabel_session(source_session, output_root, config, class_names):
    session_name = os.path.basename(source_session)
    output_session = os.path.join(output_root, session_name)
    if os.path.exists(output_session):
        raise FileExistsError(f"output session already exists: {output_session}")
    metadata_path = os.path.join(source_session, "metadata.yaml")
    with open(metadata_path, "r") as stream:
        metadata = yaml.safe_load(stream) or {}
    bag_path = resolve_path(str(metadata.get("bag_path", "")), package_root())
    detector = _load_detector(bag_path, config)

    manifest_path = os.path.join(source_session, "samples.csv")
    with open(manifest_path, "r", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"stamp", "class_index", "class_name"}
    if not required.issubset(fieldnames):
        raise ValueError(f"manifest is missing {sorted(required)}: {manifest_path}")

    os.makedirs(output_session)
    _link_payload_directories(source_session, output_session)
    turn_class_index = int(config["class_index"])
    turn_class_name = str(config["class_name"])
    changed = 0
    for row in rows:
        if detector.is_turning(float(row["stamp"])):
            if int(row["class_index"]) != turn_class_index:
                changed += 1
            row["class_index"] = str(turn_class_index)
            row["class_name"] = turn_class_name

    before_bridge = [int(row["class_index"]) for row in rows]
    bridged = bridge_short_turning_gaps(
        stamps=[float(row["stamp"]) for row in rows],
        labels=before_bridge,
        turn_class_index=turn_class_index,
        maximum_seconds=config["turning_gap_bridge_max_seconds"],
    )
    bridge_changed = sum(
        before != after for before, after in zip(before_bridge, bridged)
    )
    before_post_turn = bridged
    refined = replace_post_turn_label_islands(
        stamps=[float(row["stamp"]) for row in rows],
        labels=before_post_turn,
        turn_class_index=turn_class_index,
        maximum_seconds=config["post_turn_next_label_max_seconds"],
    )
    post_turn_changed = sum(
        before != after for before, after in zip(before_post_turn, refined)
    )
    counts = Counter()
    for row, class_index in zip(rows, refined):
        row["class_index"] = str(class_index)
        row["class_name"] = str(class_names[class_index])
        counts[class_index] += 1

    output_manifest = os.path.join(output_session, "samples.csv")
    with open(output_manifest, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata.update(
        {
            "class_names": list(class_names),
            "sample_count": len(rows),
            "class_counts": {
                name: int(counts.get(index, 0))
                for index, name in enumerate(class_names)
            },
            "turn_detection": dict(config),
            "turning_intervals": len(detector.intervals),
            "turning_samples": int(counts.get(turn_class_index, 0)),
            "turning_samples_relabelled": changed,
            "post_turn_next_label": {
                "maximum_seconds": config[
                    "post_turn_next_label_max_seconds"
                ],
                "changed_samples": post_turn_changed,
            },
            "turning_gap_bridge": {
                "maximum_seconds": config["turning_gap_bridge_max_seconds"],
                "changed_samples": bridge_changed,
            },
            "source_session_dir": os.path.abspath(source_session),
        }
    )
    with open(os.path.join(output_session, "metadata.yaml"), "w") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    print(
        f"session={session_name} samples={len(rows)} "
        f"turning={counts[turn_class_index]} "
        f"intervals={len(detector.intervals)}"
    )


def main():
    args = parse_args()
    loaded = load_collection_config(args.config_dir)
    config = loaded["turn_detection"]
    if not config["enabled"]:
        raise ValueError("turn_detection.enabled must be true")
    source_dir = os.path.abspath(os.path.expanduser(args.source_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    if source_dir == output_dir:
        raise ValueError("source and output directories must be different")
    sessions = session_directories(source_dir)
    if not sessions:
        raise ValueError(f"no dataset sessions found under {source_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for session in sessions:
        _relabel_session(
            session,
            output_dir,
            config,
            loaded["model"]["class_names"],
        )


if __name__ == "__main__":
    main()
