"""
clean_gaussians.py

Filters a trained gsplat .ply to remove floater/noise splats BEFORE it
reaches Unity, so GaussianSplatAsset creation always gets a clean input
instead of relying on manual in-editor splat deletion every time.

Targets the two dominant floater signatures:
  - near-zero opacity splats (never fully pruned by the optimizer,
    individually near-invisible but add up to haze/noise)
  - oversized-scale splats (a single Gaussian stretched huge to explain
    a handful of mismatched/noisy points -- renders as a giant blob)

Optionally also crops anything far outside the camera trajectory's
bounding volume, since real room geometry should be near where you
actually walked with the phone.

3DGS PLY fields are stored RAW (not activated) -- opacity needs a
sigmoid, scale needs an exp, to get real-world interpretable values.
This script reads raw, computes activated values for thresholding, but
writes back the ORIGINAL raw fields for the kept splats (so Unity's
importer, which expects raw gsplat PLY format, is unaffected).

Usage:
    python clean_gaussians.py --base-dir <capture_dir> --max-steps 30000
"""
import os
import json
import struct
import collections
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", required=True)
parser.add_argument("--max-steps", type=int, default=30000)
parser.add_argument("--opacity-thresh", type=float, default=0.05,
                     help="Sigmoid-activated opacity below this is dropped. "
                          "0.05 is conservative; raise to 0.1-0.15 for a more "
                          "aggressive clean if haze persists.")
parser.add_argument("--scale-percentile", type=float, default=99.5,
                     help="Splats whose max scale axis (exp-activated) exceeds "
                          "this percentile are dropped as oversized blobs.")
parser.add_argument("--camera-margin", type=float, default=0.3,
                     help="Optional: also drop splats farther than this distance "
                          "(normalized scene scale) outside the camera trajectory's "
                          "bounding box. Set to 0 to disable this crop entirely.")
parser.add_argument("--suffix", default="_clean",
                     help="Output filename suffix, e.g. point_cloud_29999_clean.ply")
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
print(f"Loaded {n_total} splats, fields: {field_names}")

xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)

# ---- opacity: gsplat stores raw logit, sigmoid to get real 0-1 opacity ----
assert "opacity" in field_names, "No 'opacity' field -- unexpected PLY schema"
opacity_raw = data["opacity"].astype(np.float64)
opacity_activated = 1.0 / (1.0 + np.exp(-opacity_raw))

# ---- scale: gsplat stores raw log-scale, exp to get real linear scale ----
scale_fields = [f for f in field_names if f.startswith("scale_")]
assert len(scale_fields) == 3, f"Expected 3 scale fields, got {scale_fields}"
scale_raw = np.stack([data[f] for f in scale_fields], axis=1).astype(np.float64)
scale_activated = np.exp(scale_raw)
scale_max_axis = scale_activated.max(axis=1)

keep = np.ones(n_total, dtype=bool)

# ---- filter 1: low opacity ----
low_opacity = opacity_activated < args.opacity_thresh
keep &= ~low_opacity
print(f"Dropping {low_opacity.sum()} splats below opacity {args.opacity_thresh}")

# ---- filter 2: oversized scale (blob floaters) ----
scale_cutoff = np.percentile(scale_max_axis, args.scale_percentile)
oversized = scale_max_axis > scale_cutoff
keep &= ~oversized
print(f"Dropping {oversized.sum()} splats above scale percentile "
      f"{args.scale_percentile} (cutoff={scale_cutoff:.4f})")

# ---- filter 3 (optional): outside camera trajectory bounding box ----
if args.camera_margin > 0:
    Image = collections.namedtuple("Image", ["tvec"])

    def read_images_binary_tvec_only(path):
        tvecs = []
        with open(path, "rb") as f:
            num = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num):
                struct.unpack("<i", f.read(4))       # image_id
                struct.unpack("<4d", f.read(32))      # qvec, unused
                tvec = struct.unpack("<3d", f.read(24))
                struct.unpack("<i", f.read(4))        # camera_id
                name_bytes = bytearray()
                while True:
                    c = f.read(1)
                    if c == b"\x00":
                        break
                    name_bytes += c
                n_pts = struct.unpack("<Q", f.read(8))[0]
                f.read(16 * n_pts)
                f.read(8 * n_pts)
                tvecs.append(tvec)
        return np.array(tvecs)

    norm_path = os.path.join(OUTPUT_DIR, "normalization.json")
    images_bin = os.path.join(SPARSE_DIR, "images.bin")
    if os.path.exists(norm_path) and os.path.exists(images_bin):
        with open(norm_path) as f:
            T = np.array(json.load(f)["transform"])
        raw_tvecs = read_images_binary_tvec_only(images_bin)
        # tvec is world-to-camera translation; camera CENTER (world space)
        # requires the rotation too -- but for a bounding-box margin, using
        # -tvec as a rough proxy is good enough since this is just a coarse
        # spatial gate, not a precise position (skip full R,t inverse here).
        # For accuracy, reuse frame_c2w logic from build_object_meshes.py
        # if this crop needs to be tighter.
        cam_pts_h = np.hstack([-raw_tvecs, np.ones((len(raw_tvecs), 1))])
        cam_pts_world = (T @ cam_pts_h.T).T[:, :3]

        cam_min = cam_pts_world.min(axis=0) - args.camera_margin
        cam_max = cam_pts_world.max(axis=0) + args.camera_margin
        outside_bbox = np.any((xyz < cam_min) | (xyz > cam_max), axis=1)
        keep &= ~outside_bbox
        print(f"Dropping {outside_bbox.sum()} splats outside camera trajectory "
              f"bbox (margin={args.camera_margin})")
    else:
        print("Skipping camera-trajectory crop -- normalization.json or "
              "images.bin not found (run compute_transform.py first)")

n_kept = keep.sum()
print(f"\nKeeping {n_kept}/{n_total} splats ({100*n_kept/n_total:.1f}%)")
assert n_kept > 0, "Every splat got filtered out -- thresholds are too aggressive"

# ---- write cleaned PLY, preserving original raw schema exactly ----
cleaned_data = data[keep]
cleaned_element = PlyElement.describe(cleaned_data, "vertex")

out_name = f"point_cloud_{args.max_steps - 1}{args.suffix}.ply"
out_path = os.path.join(OUTPUT_DIR, "ply", out_name)
PlyData([cleaned_element], text=False).write(out_path)

print(f"\nWrote cleaned PLY: {out_path}")
print("Point this at Unity's Gaussian Splat Creator (Input PLY/SPZ File)")
print("instead of the raw trained point_cloud_N.ply.")