#!/usr/bin/env python3
"""Adds a bird's-eye-view (BEV) occupancy grid to each dataset session.

Unlike add_bev_to_dataset.py, which keeps only the single nearest point in
each lateral bin (throwing away every other point, the height axis, and any
notion of density), this bins EVERY filtered point into a fixed
forward x lateral grid and records, per cell, how many points fell in it
and their mean height. Nothing is discarded except the discretization
itself, so it is the "use all the points" representation -- the same first
step PointPillars takes before its learned per-pillar encoder.

The distance/height filtering is deliberately kept identical to
extract_nearest_obstacle_points (which CARE uses live) so the only
difference under test is the aggregation.

Originally: Adds a bird's-eye-view (BEV) obstacle scan to each dataset session,
reusing the exact same depth->point-cloud->BEV pipeline CARE uses live
(unidepth_ros.depth_estimator: filter_metric_point_cloud,
transform_point_cloud, extract_nearest_obstacle_points). The camera->robot
transform is static (an all-fixed-joint chain from base_footprint to
camera_rgb_optical_frame in the robot URDF), so it is resolved once via TF
at startup and reused for every frame rather than looked up per-frame.

Mirrors add_depth_to_dataset.py's bag-replay/timestamp-matching structure,
but was not that useful without depth already generated -- this script
also re-runs UniDepth (it needs prediction.points_camera, which
add_depth_to_dataset.py discards) rather than reusing saved depth.
"""
import argparse
import bisect
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import numpy as np
import rosbag
import rospy
import tf2_ros
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
from unidepth_ros.depth_estimator import (
    UniDepthV2Small,
    extract_nearest_obstacle_points,
    filter_metric_point_cloud,
    transform_point_cloud,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=None)
    parser.add_argument(
        "--camera-optical-frame", default="camera_rgb_optical_frame"
    )
    parser.add_argument("--robot-frame", default="base_footprint")
    parser.add_argument("--tf-timeout-seconds", type=float, default=5.0)
    # The camera chain is all fixed joints, so the transform can be supplied
    # directly instead of resolved over TF -- which needs a live
    # robot_state_publisher. For orne_gamma
    # (orne_description/urdf/gamma/orne_gamma.urdf.xacro) it is
    #   --camera-translation 0.227090108 0.0 0.373559849
    #   --camera-rotation -0.530019283 0.529646723 -0.468059016 0.468481233
    parser.add_argument("--camera-translation", type=float, nargs=3, default=None)
    parser.add_argument(
        "--camera-rotation", type=float, nargs=4, default=None,
        help="camera->robot quaternion as x y z w",
    )
    parser.add_argument("--minimum-depth-m", type=float, default=0.1)
    parser.add_argument("--maximum-depth-m", type=float, default=10.0)
    parser.add_argument("--minimum-forward-m", type=float, default=0.1)
    parser.add_argument("--maximum-forward-m", type=float, default=8.0)
    parser.add_argument("--minimum-height-m", type=float, default=0.05)
    parser.add_argument("--maximum-height-m", type=float, default=1.6)
    parser.add_argument("--map-width-m", type=float, default=8.0)
    # Square cells: the forward span (minimum_forward_m..maximum_forward_m)
    # and the lateral span (map_width_m) are both ~8 m, so equal bin counts
    # give ~0.125 m cells in both axes. 16x32 (the first attempt) made
    # forward cells twice as coarse as lateral ones for no reason.
    parser.add_argument("--lateral-bins", type=int, default=64)
    parser.add_argument("--forward-bins", type=int, default=64)
    parser.add_argument("--border-margin-ratio", type=float, default=0.02)
    parser.add_argument("--output-subdir", default="bev_grid")
    parser.add_argument("--manifest-column", default="bev_grid_filename")
    if argv is None:
        argv = sys.argv
    return parser.parse_args(rospy.myargv(argv=argv)[1:])



def bev_occupancy_grid(robot_point_cloud, valid_depth_mask, args):
    """Bin every filtered point into a (forward_bins, lateral_bins, 2) grid.

    Channel 0 is the point count in the cell, channel 1 the mean height of
    those points (0 where the cell is empty). Robot coordinates are x
    forward, y left, z up; the filters match
    extract_nearest_obstacle_points so only the aggregation differs.
    """
    points = np.asarray(robot_point_cloud, dtype=np.float32)
    mask = np.asarray(valid_depth_mask, dtype=bool)
    forward, left, height = points[..., 0], points[..., 1], points[..., 2]
    lateral_limit = args.map_width_m / 2.0
    keep = (
        mask
        & (forward >= args.minimum_forward_m)
        & (forward <= args.maximum_forward_m)
        & (left >= -lateral_limit)
        & (left <= lateral_limit)
        & (height >= args.minimum_height_m)
        & (height <= args.maximum_height_m)
    )
    rows, cols = keep.shape
    margin_y = int(round(rows * args.border_margin_ratio))
    margin_x = int(round(cols * args.border_margin_ratio))
    if margin_y > 0:
        keep[:margin_y] = False
        keep[rows - margin_y :] = False
    if margin_x > 0:
        keep[:, :margin_x] = False
        keep[:, cols - margin_x :] = False

    grid = np.zeros((args.forward_bins, args.lateral_bins, 2), dtype=np.float32)
    selected = np.flatnonzero(keep)
    if selected.size == 0:
        return grid
    f = forward.ravel()[selected]
    l = left.ravel()[selected]
    z = height.ravel()[selected]
    span = args.maximum_forward_m - args.minimum_forward_m
    fi = np.minimum(
        np.floor((f - args.minimum_forward_m) / span * args.forward_bins).astype(np.int32),
        args.forward_bins - 1,
    )
    li = np.minimum(
        np.floor((l + lateral_limit) / args.map_width_m * args.lateral_bins).astype(np.int32),
        args.lateral_bins - 1,
    )
    flat = fi * args.lateral_bins + li
    counts = np.bincount(flat, minlength=args.forward_bins * args.lateral_bins)
    sums = np.bincount(flat, weights=z, minlength=args.forward_bins * args.lateral_bins)
    grid[..., 0] = counts.reshape(args.forward_bins, args.lateral_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_height = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    grid[..., 1] = mean_height.reshape(args.forward_bins, args.lateral_bins)
    return grid


def _load_manifest(session_dir):
    path = os.path.join(session_dir, "samples.csv")
    with open(path, "r", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "stamp" not in rows[0]:
        raise ValueError(f"manifest has no timestamped samples: {path}")
    return path, rows


def _write_manifest(path, rows, manifest_column="bev_grid_filename"):
    fieldnames = list(rows[0].keys())
    if manifest_column not in fieldnames:
        fieldnames.insert(1, manifest_column)
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
    if not bag_path or not image_topic:
        raise ValueError(
            f"session metadata must contain bag_path and image_topic: {path}"
        )
    return bag_path, image_topic


def resolve_static_camera_transform(camera_optical_frame, robot_frame, timeout_seconds):
    buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(buffer)
    rospy.sleep(1.5)
    transform = buffer.lookup_transform(
        robot_frame, camera_optical_frame, rospy.Time(0), rospy.Duration(timeout_seconds)
    )
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    return (translation.x, translation.y, translation.z), (
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    )


def add_bev_to_session(
    session_dir, estimator, tolerance_seconds, translation, quaternion, args
):
    manifest_path, rows = _load_manifest(session_dir)
    bag_path, image_topic = _session_metadata(session_dir)
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"source bag was not found: {bag_path}")
    targets = sorted((float(row["stamp"]), index) for index, row in enumerate(rows))
    target_stamps = [item[0] for item in targets]
    unmatched = set(range(len(rows)))
    bev_dir = os.path.join(session_dir, args.output_subdir)
    os.makedirs(bev_dir, exist_ok=True)
    bridge = CvBridge()
    progress = tqdm(total=len(rows), desc=os.path.basename(session_dir), unit="frame")
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
            camera_points, valid_depth_mask = filter_metric_point_cloud(
                prediction.points_camera,
                prediction.depth_meters,
                minimum_depth_m=args.minimum_depth_m,
                maximum_depth_m=args.maximum_depth_m,
            )
            robot_points = transform_point_cloud(camera_points, translation, quaternion)
            grid = bev_occupancy_grid(robot_points, valid_depth_mask, args)
            relative_path = os.path.join(args.output_subdir, f"{match:06d}.npy")
            np.save(os.path.join(session_dir, relative_path), grid, allow_pickle=False)
            rows[match][args.manifest_column] = relative_path
            unmatched.remove(match)
            progress.update(1)
            if not unmatched:
                break
    progress.close()
    if unmatched:
        raise ValueError(
            f"could not match {len(unmatched)} dataset stamps in {bag_path}"
        )
    _write_manifest(manifest_path, rows, args.manifest_column)


def main():
    args = parse_args()
    rospy.init_node("corridor_dataset_bev_generator")
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

    rospy.loginfo(
        "resolving static transform %s -> %s",
        args.camera_optical_frame,
        args.robot_frame,
    )
    if args.camera_translation is not None and args.camera_rotation is not None:
        translation = tuple(args.camera_translation)
        quaternion = tuple(args.camera_rotation)
    else:
        translation, quaternion = resolve_static_camera_transform(
            args.camera_optical_frame, args.robot_frame, args.tf_timeout_seconds
        )
    rospy.loginfo("translation=%s quaternion=%s", translation, quaternion)

    dataset_root = resolve_path(config["paths"]["dataset_dir"], package_root())
    sessions = []
    for dataset_type in generation["dataset_types"]:
        sessions.extend(session_directories(os.path.join(dataset_root, dataset_type)))
    selected_names = set(generation["session_names"])
    if selected_names:
        sessions = [
            session
            for session in sessions
            if os.path.basename(session) in selected_names
        ]
    if not sessions:
        raise ValueError("no dataset sessions selected for BEV generation")
    for session in sessions:
        add_bev_to_session(
            session,
            estimator,
            generation["stamp_tolerance_seconds"],
            translation,
            quaternion,
            args,
        )
    rospy.loginfo("added BEV scans to %d dataset session(s)", len(sessions))


if __name__ == "__main__":
    main()
