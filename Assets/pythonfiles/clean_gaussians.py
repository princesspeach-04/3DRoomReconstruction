"""
clean_gaussians.py

Filters a trained gsplat .ply using FOUR signals:
  1. opacity        - near-invisible splats the optimizer never pruned
  2. scale          - oversized blob splats explaining noisy matches
  3. view-count     - splats seen from too few distinct camera angles
  4. largest-cluster (optional) - for single-subject captures, keep ONLY
                       the biggest spatially-connected blob, discard
                       every smaller floater fragment entirely

Usage (room-scale, keep everything that passes quality filters):
    python clean_gaussians.py --base-dir <dir> --max-steps 30000

Usage (single-object/subject capture, e.g. "just this table+chairs"):
    python clean_gaussians.py --base-dir <dir> --max-steps 30000 --keep-largest-cluster
"""
import os
import json
import struct
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", required=True)
parser.add_argument("--max-steps", type=int, default=30000)
parser.add_argument("--opacity-thresh", type=float, default=0.05)
parser.add_argument("--scale-percentile", type=float, default=98.0)
parser.add_argument("--min-view-count", type=int, default=3,
                     help="Splat must be seen in at least this many camera "
                          "frames to survive. Main floater killer -- raise to "
                          "5-8 for a much more aggressive clean.")
parser.add_argument("--view-check-stride", type=int, default=1,
                     help="Check every Nth frame for speed. 2 halves runtime "
                          "with negligible accuracy loss on video captures.")
parser.add_argument("--radius-outlier-nb", type=int, default=16,
                     help="Min neighbors required nearby. 0 disables this pass.")
parser.add_argument("--radius-outlier-radius", type=float, default=0.05)
parser.add_argument("--keep-largest-cluster", action="store_true",
                     help="After all other filters, DBSCAN-cluster what's left "
                          "and keep ONLY the largest connected blob. Use for "
                          "single-object/single-subject captures.")
parser.add_argument("--cluster-eps", type=float, default=0.08,
                     help="DBSCAN eps for the largest-cluster pass -- same "
                          "ballpark as check.py's sweep, normalized scene scale.")
parser.add_argument("--cluster-min-points", type=int, default=50)
parser.add_argument("--suffix", default="_clean")
args = parser.parse_args()

BASE_DIR = args.base_dir
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SPARSE_DIR = os.path.join(BASE_DIR, "sparse", "0")

ply_path = os.path.join(OUTPUT_DIR, "ply", f"point_cloud_{args.max_steps - 1}.ply")
assert os.path.exists(ply_path), f"Trained PLY not found at {ply_path}"

print(f"Loading {ply_path} ...")
ply = PlyData.read(ply_path)
v = ply["vertex"]
data = v.data
field_names = data.dtype.names
n_total = len(data)
print(f"Loaded {n_total} splats")

xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)

opacity_activated = 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float64)))
scale_fields = [f for f in field_names if f.startswith("scale_")]
scale_activated = np.exp(np.stack([data[f] for f in scale_fields], axis=1).astype(np.float64))
scale_max_axis = scale_activated.max(axis=1)

keep = np.ones(n_total, dtype=bool)

low_opacity = opacity_activated < args.opacity_thresh
keep &= ~low_opacity
print(f"Dropping {low_opacity.sum()} splats below opacity {args.opacity_thresh}")

scale_cutoff = np.percentile(scale_max_axis, args.scale_percentile)
oversized = scale_max_axis > scale_cutoff
keep &= ~oversized
print(f"Dropping {oversized.sum()} splats above scale percentile "
      f"{args.scale_percentile} (cutoff={scale_cutoff:.4f})")

# ---- multi-view visibility count ----
from colmap_io import read_images_binary, read_cameras_binary, qvec_to_rotmat

norm_path = os.path.join(OUTPUT_DIR, "normalization.json")
images_bin = os.path.join(SPARSE_DIR, "images.bin")
cameras_bin = os.path.join(SPARSE_DIR, "cameras.bin")
assert os.path.exists(norm_path), "Run compute_transform.py first"

with open(norm_path) as f:
    T = np.array(json.load(f)["transform"])
images = read_images_binary(images_bin)
cameras = read_cameras_binary(cameras_bin)

frame_names = list(images.keys())[::args.view_check_stride]
print(f"\nChecking multi-view visibility across {len(frame_names)} frames "
      f"(stride={args.view_check_stride})...")

view_counts = np.zeros(n_total, dtype=np.int32)
pts_h = np.hstack([xyz, np.ones((n_total, 1))])

for fname in frame_names:
    img = images[fname]
    camera = cameras[img.camera_id]
    R = qvec_to_rotmat(img.qvec)
    t = np.array(img.tvec)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)
    c2w_t = T @ c2w
    scale = np.linalg.norm(c2w_t[:3, 0])
    c2w_t[:3, :3] /= scale
    w2c_t = np.linalg.inv(c2w_t)

    cam_pts = (w2c_t @ pts_h.T).T[:, :3]
    in_front = cam_pts[:, 2] > 1e-6
    u = camera.fx * cam_pts[:, 0] / np.clip(cam_pts[:, 2], 1e-6, None) + camera.cx
    vv = camera.fy * cam_pts[:, 1] / np.clip(cam_pts[:, 2], 1e-6, None) + camera.cy
    visible = in_front & (u >= 0) & (u < camera.width) & (vv >= 0) & (vv < camera.height)
    view_counts += visible

low_view_count = view_counts < args.min_view_count
keep &= ~low_view_count
print(f"Dropping {low_view_count.sum()} splats seen in fewer than "
      f"{args.min_view_count} camera frames")

n_after_stage1 = keep.sum()
print(f"\n{n_after_stage1}/{n_total} splats survive opacity+scale+view-count "
      f"({100*n_after_stage1/n_total:.1f}%)")
assert n_after_stage1 > 1000, "Filters too aggressive -- loosen thresholds"

# ---- radius-outlier removal ----
import open3d as o3d

if args.radius_outlier_nb > 0:
    kept_idx = np.where(keep)[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[kept_idx])
    _, inlier_idx = pcd.remove_radius_outlier(
        nb_points=args.radius_outlier_nb, radius=args.radius_outlier_radius)
    inlier_mask = np.zeros(len(kept_idx), dtype=bool)
    inlier_mask[inlier_idx] = True
    dropped_by_radius = kept_idx[~inlier_mask]
    keep[dropped_by_radius] = False
    print(f"Dropping {len(dropped_by_radius)} more splats via radius-outlier "
          f"removal (isolated points, few neighbors within "
          f"{args.radius_outlier_radius})")

# ---- keep only largest connected cluster (single-subject captures) ----
if args.keep_largest_cluster:
    kept_idx = np.where(keep)[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[kept_idx])
    labels = np.array(pcd.cluster_dbscan(
        eps=args.cluster_eps, min_points=args.cluster_min_points, print_progress=False))

    if labels.max() >= 0:
        sizes = [(labels == i).sum() for i in range(labels.max() + 1)]
        largest_label = int(np.argmax(sizes))
        print(f"\nFound {labels.max()+1} clusters, sizes={sorted(sizes, reverse=True)[:8]}")
        print(f"Keeping only largest cluster (label {largest_label}, "
              f"{sizes[largest_label]} points)")
        cluster_keep_mask = labels == largest_label
        drop_idx = kept_idx[~cluster_keep_mask]
        keep[drop_idx] = False
    else:
        print("WARNING: no clusters found at this eps, skipping largest-cluster filter")

n_kept = keep.sum()
print(f"\nFinal: keeping {n_kept}/{n_total} splats ({100*n_kept/n_total:.1f}%)")
assert n_kept > 500, "Every filter combined removed almost everything -- loosen thresholds"

cleaned_data = data[keep]
cleaned_element = PlyElement.describe(cleaned_data, "vertex")
out_name = f"point_cloud_{args.max_steps - 1}{args.suffix}.ply"
out_path = os.path.join(OUTPUT_DIR, "ply", out_name)
PlyData([cleaned_element], text=False).write(out_path)

print(f"\nWrote cleaned PLY: {out_path}")
print("Point Unity's Gaussian Splat Creator at this file, NOT the raw one.")