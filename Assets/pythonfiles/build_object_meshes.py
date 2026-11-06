"""
build_object_meshes.py   (v2: visibility-aware voting)

Multi-class object mesh extraction. For every class present in
detections_seg.json (or every class in --target-classes if given):

  1. Projects every trained Gaussian-Splat point into every camera frame that
     has a detection.
  2. NEW: keeps only splats that are the FIRST VISIBLE SURFACE along their camera
     ray (a coarse z-buffer made from the splats themselves). Before this, every
     splat BEHIND an object (the floor under a laptop, the whole room behind a
     chair) also fell inside the object's 2D mask and got a vote, which produced
     scene-sized "objects" and phantom clusters on the floor.
  3. A visible splat that lands inside a YOLO-SEG mask for that class in enough
     DIFFERENT frames gets a vote.
  4. DBSCAN-clusters voted points into individual object instances.
  5. Reconstructs a real mesh per cluster with Open3D alpha-shape.
  6. Records quality metrics (triangle_count, bbox_aspect_ratio) per object.

Requires (already produced earlier in your pipeline):
  - output/ply/point_cloud_<N>_clean.ply      (cleaned splat, from clean_gaussians.py)
  - sparse/0/images.bin, cameras.bin           (raw COLMAP poses/intrinsics)
  - output/normalization.json                  (from compute_transform.py)
  - segmentation/detections_seg.json + masks/  (from run_yolo_segmentation.py)

Usage:
    python build_object_meshes.py --base-dir <capture_dir> --max-steps 30000
    python build_object_meshes.py --base-dir <capture_dir> --max-steps 30000 --target-classes chair laptop
    python build_object_meshes.py ... --no-visibility        (old behaviour, for comparison)
"""
import os
import json
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy import ndimage

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", required=True)
parser.add_argument("--max-steps", type=int, default=7000,
                     help="Must match the trained .ply you're targeting -- "
                          "used to find point_cloud_<max_steps-1>.ply")
parser.add_argument("--ply-suffix", default="_clean",
                     help="Which splat file to use: point_cloud_<N><suffix>.ply. "
                          "Default '_clean' (floaters already removed). Use '' for the raw PLY.")
parser.add_argument("--target-classes", nargs="+", default=None,
                     help="Classes to reconstruct. Defaults to every class "
                          "present in detections_seg.json.")
parser.add_argument("--min-frame-votes", type=int, default=3,
                     help="A visible point must land inside the mask in at least "
                          "this many distinct frames to count as 'on the object'.")
parser.add_argument("--min-vote-ratio", type=float, default=0.3,
                     help="A point must ALSO be inside the class mask in at least this share of the frames "
                          "(where that class was detected) in which it is visible. Stops floors / big "
                          "objects from merging with the chair next to them. 0 = off (old behaviour).")
parser.add_argument("--dbscan-eps", type=float, default=0.05)
parser.add_argument("--dbscan-min-points", type=int, default=30)
parser.add_argument("--alpha", type=float, default=0.05,
                     help="Open3D alpha-shape parameter -- smaller = tighter surface.")
parser.add_argument("--min-cluster-size", type=int, default=80,
                     help="Discard DBSCAN clusters smaller than this.")
parser.add_argument("--no-visibility", action="store_true",
                     help="Disable the first-visible-surface test (old, leaky voting).")
parser.add_argument("--vis-tol", type=float, default=0.03,
                     help="Depth tolerance (scene units) for 'same surface as the front-most splat'.")
parser.add_argument("--vis-opacity", type=float, default=0.3,
                     help="Only splats at least this opaque can occlude others in the z-buffer.")
parser.add_argument("--vis-grid-div", type=int, default=8,
                     help="z-buffer cell size in pixels.")
args = parser.parse_args()

BASE_DIR = args.base_dir
SPARSE_DIR = os.path.join(BASE_DIR, "sparse", "0")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MESH_DIR = os.path.join(OUTPUT_DIR, "meshes")
SEG_DIR = os.path.join(BASE_DIR, "segmentation")
MASKS_DIR = os.path.join(SEG_DIR, "masks")
print(f"Using BASE_DIR: {BASE_DIR}")
os.makedirs(MESH_DIR, exist_ok=True)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import read_images_binary, read_cameras_binary, qvec_to_rotmat


# ---- load everything ----
ply_path = os.path.join(OUTPUT_DIR, "ply", f"point_cloud_{args.max_steps - 1}{args.ply_suffix}.ply")
if not os.path.exists(ply_path):
    raw = os.path.join(OUTPUT_DIR, "ply", f"point_cloud_{args.max_steps - 1}.ply")
    print(f"WARNING: {ply_path} not found, falling back to RAW ply (floaters included!)")
    ply_path = raw
assert os.path.exists(ply_path), f"Trained PLY not found at {ply_path}"
print(f"Using splat file: {ply_path}")
ply = PlyData.read(ply_path)
v = ply["vertex"]
points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
if "opacity" in v.data.dtype.names:
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
else:
    opacity = np.ones(len(points))
solid = opacity >= args.vis_opacity
print(f"Loaded {len(points)} trained splat points ({int(solid.sum())} opaque enough to occlude)")

images = read_images_binary(os.path.join(SPARSE_DIR, "images.bin"))
cameras = read_cameras_binary(os.path.join(SPARSE_DIR, "cameras.bin"))
for c in cameras.values():
    print(f"Camera {c.id} [{c.model}] {c.width}x{c.height}  fx={c.fx:.1f} fy={c.fy:.1f} cx={c.cx:.1f} cy={c.cy:.1f}")
print(f"Loaded {len(images)} camera poses")


def project(cam_pts, cam, mask_shape):
    """Project camera-space points; returns (ui, vi, valid) in MASK pixel coords
    (handles masks whose resolution differs from the COLMAP image size)."""
    z = np.clip(cam_pts[:, 2], 1e-6, None)
    u = cam.fx * cam_pts[:, 0] / z + cam.cx
    v_ = cam.fy * cam_pts[:, 1] / z + cam.cy
    ok = (cam_pts[:, 2] > 1e-6) & (u >= 0) & (u < cam.width) & (v_ >= 0) & (v_ < cam.height)
    mh, mw = mask_shape
    ui = np.clip((u * mw / cam.width).astype(np.int32), 0, mw - 1)
    vi = np.clip((v_ * mh / cam.height).astype(np.int32), 0, mh - 1)
    return ui, vi, ok


def first_visible(cam_pts, cam):
    """Boolean per point: in the image AND not hidden behind a nearer opaque splat.
    Coarse z-buffer (cells of --vis-grid-div px; empty cells are filled from their
    neighbours so gaps between splat centres don't let background leak through)."""
    z = cam_pts[:, 2]
    zs = np.clip(z, 1e-6, None)
    u = cam.fx * cam_pts[:, 0] / zs + cam.cx
    vv = cam.fy * cam_pts[:, 1] / zs + cam.cy
    ok = (z > 1e-6) & (u >= 0) & (u < cam.width) & (vv >= 0) & (vv < cam.height)
    if args.no_visibility:
        return ok
    d = float(args.vis_grid_div)
    gw, gh = max(int(cam.width / d), 2), max(int(cam.height / d), 2)
    gx = np.clip((u / d).astype(np.int64), 0, gw - 1)
    gy = np.clip((vv / d).astype(np.int64), 0, gh - 1)
    sel = ok & solid
    zb = np.full((gh, gw), np.inf, dtype=np.float32)
    np.minimum.at(zb, (gy[sel], gx[sel]), z[sel].astype(np.float32))
    # close gaps ONLY in empty cells (a min-filter over all cells would smear the near depth
    # over steep side faces and hide visible surfaces such as chair legs / object sides)
    zb = np.where(np.isinf(zb), ndimage.minimum_filter(zb, size=3), zb)
    return ok & (z <= zb[gy, gx] + args.vis_tol + 0.02 * z)


with open(os.path.join(OUTPUT_DIR, "normalization.json")) as f:
    T = np.array(json.load(f)["transform"])

with open(os.path.join(SEG_DIR, "detections_seg.json")) as f:
    detections = json.load(f)

frame_c2w = {}
frame_cam = {}
for img in images.values():
    R = qvec_to_rotmat(img.qvec)
    t = np.array(img.tvec)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)
    c2w_t = T @ c2w
    scale = np.linalg.norm(c2w_t[:3, 0])
    c2w_t[:3, :3] /= scale
    frame_c2w[img.name] = c2w_t
    frame_cam[img.name] = cameras[img.camera_id]

all_classes_present = sorted({e["class"] for ents in detections.values() for e in ents})
target_classes = args.target_classes if args.target_classes else all_classes_present
print(f"\nProcessing classes: {target_classes}")

# ---- pass 1: vote, all classes at once (each frame is projected only once) ----
votes = {c: np.zeros(len(points), dtype=np.int16) for c in target_classes}
seen = {c: np.zeros(len(points), dtype=np.int16) for c in target_classes}
frames_with = {c: [] for c in target_classes}
for fname, ents in detections.items():
    for c in {e["class"] for e in ents}:
        if c in votes:
            frames_with[c].append(fname)
frames_needed = sorted({f for fl in frames_with.values() for f in fl})
print(f"Voting over {len(frames_needed)} frames "
      f"({'first-visible-surface only' if not args.no_visibility else 'NO visibility test'}) ...")

pts32 = points.astype(np.float32)
for fi, fname in enumerate(frames_needed):
    if fname not in frame_c2w:
        continue
    npz_path = os.path.join(MASKS_DIR, os.path.splitext(fname)[0] + ".npz")
    if not os.path.exists(npz_path):
        continue
    w2c = np.linalg.inv(frame_c2w[fname])
    cam_pts = pts32 @ w2c[:3, :3].T.astype(np.float32) + w2c[:3, 3].astype(np.float32)
    cam = frame_cam[fname]
    vis = first_visible(cam_pts, cam)
    masks = np.load(npz_path)
    ent_list = detections[fname]
    if ent_list:
        shape0 = masks[ent_list[0]["mask_key"]].shape
        _u, _v, ok_img = project(cam_pts, cam, shape0)
        for c in {e["class"] for e in ent_list}:
            if c in seen:
                seen[c][vis & ok_img] += 1
    for ent in ent_list:
        c = ent["class"]
        if c not in votes:
            continue
        mask = masks[ent["mask_key"]]
        ui, vi, ok = project(cam_pts, cam, mask.shape)
        hit = vis & ok & mask[vi, ui].astype(bool)
        votes[c][hit] += 1
    if (fi + 1) % 50 == 0 or fi == len(frames_needed) - 1:
        print(f"  voted {fi + 1}/{len(frames_needed)} frames")

manifest = []
pts_h_all = None

for target_class in target_classes:
    print(f"\n{'='*50}")
    print(f"Class: {target_class}")
    print(f"{'='*50}")

    frames_with_target = frames_with[target_class]
    print(f"{len(frames_with_target)} frames contain '{target_class}' detections")
    if len(frames_with_target) == 0:
        continue

    v_c, s_c = votes[target_class].astype(np.float32), np.maximum(seen[target_class], 1).astype(np.float32)
    base_ok = v_c >= args.min_frame_votes
    ratio = v_c / s_c
    row = "  ".join(f"r>={r:.1f}:{int((base_ok & (ratio >= r)).sum())}" for r in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6))
    print(f"points surviving per vote ratio (>= {args.min_frame_votes} votes): {row}")
    keep_mask = base_ok & (ratio >= args.min_vote_ratio)
    voted = points[keep_mask]
    print(f"{len(voted)} points kept (>= {args.min_frame_votes} votes and ratio >= {args.min_vote_ratio}) for '{target_class}'")
    if len(voted) < args.dbscan_min_points:
        print(f"  too few voted points for '{target_class}' (need >= {args.dbscan_min_points}), skipping")
        continue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voted)
    labels = np.array(pcd.cluster_dbscan(
        eps=args.dbscan_eps, min_points=args.dbscan_min_points, print_progress=False))
    n_clusters = labels.max() + 1
    print(f"DBSCAN found {n_clusters} raw clusters")

    kept = 0
    for cluster_id in range(n_clusters):
        cluster_pts = voted[labels == cluster_id]
        if len(cluster_pts) < args.min_cluster_size:
            continue

        centroid = cluster_pts.mean(axis=0)
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts - centroid)
        cluster_pcd, _ = cluster_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        cluster_pcd.estimate_normals()

        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                cluster_pcd, args.alpha)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.compute_vertex_normals()
        except Exception as e:
            print(f"  cluster {cluster_id}: mesh reconstruction failed ({e}), skipping")
            continue

        if len(mesh.triangles) == 0:
            print(f"  cluster {cluster_id}: alpha-shape produced no triangles, "
                  f"try a larger --alpha, skipping")
            continue

        bbox = cluster_pcd.get_axis_aligned_bounding_box()
        dims = (bbox.get_max_bound() - bbox.get_min_bound()).tolist()

        sorted_dims = sorted(dims, reverse=True)
        bbox_aspect_ratio = sorted_dims[0] / max(sorted_dims[-1], 1e-6)
        triangle_count = len(mesh.triangles)
        point_count = len(cluster_pts)

        obj_name = f"{target_class}_{kept}.obj"
        obj_path = os.path.join(MESH_DIR, obj_name)
        points_path = os.path.join(MESH_DIR, f"{target_class}_{kept}_points.npy")
        np.save(points_path, cluster_pts.astype(np.float32))   # used by generate_replacement.py to find clean views
        mesh.triangle_normals = o3d.utility.Vector3dVector()  # OBJ cannot store them
        o3d.io.write_triangle_mesh(obj_path, mesh)

        avg_confidence = float(np.mean([
            e["confidence"] for f in frames_with_target for e in detections[f]
            if e["class"] == target_class
        ]))

        # ---- best-matching frame for THIS instance (kept for compatibility;
        # generate_replacement.py now selects its own views) ----
        best_frame, best_mask_key, best_overlap, best_conf = None, None, -1, 0.0
        cpts_h = np.hstack([cluster_pts, np.ones((len(cluster_pts), 1))])

        for fname in frames_with_target:
            if fname not in frame_c2w:
                continue
            w2c_f = np.linalg.inv(frame_c2w[fname])
            cam_pts_f = (w2c_f @ cpts_h.T).T[:, :3]

            npz_path_f = os.path.join(MASKS_DIR, os.path.splitext(fname)[0] + ".npz")
            if not os.path.exists(npz_path_f):
                continue
            masks_f = np.load(npz_path_f)

            for ent in detections[fname]:
                if ent["class"] != target_class:
                    continue
                mask_f = masks_f[ent["mask_key"]]
                ui_f, vi_f, ok_f = project(cam_pts_f, frame_cam[fname], mask_f.shape)
                overlap = int((ok_f & mask_f[vi_f, ui_f].astype(bool)).sum())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_frame = fname
                    best_mask_key = ent["mask_key"]
                    best_conf = ent["confidence"]

        manifest.append({
            "id": f"{target_class}_{kept}",
            "class": target_class,
            "position": centroid.tolist(),
            "size": dims,
            "point_count": point_count,
            "triangle_count": triangle_count,
            "best_frame": best_frame,
            "best_mask_key": best_mask_key,
            "best_frame_overlap": best_overlap,
            "best_frame_confidence": best_conf,
            "bbox_aspect_ratio": round(bbox_aspect_ratio, 3),
            "confidence": avg_confidence,
            "mass": 5.0,
            "drag": 0.5,
            "mesh_file": obj_path,
            "points_file": points_path,
            "source_ply": os.path.basename(ply_path),
        })
        print(f"  cluster {cluster_id} -> {obj_name}: "
              f"{point_count} pts, {triangle_count} triangles, "
              f"aspect={bbox_aspect_ratio:.2f}, size={np.round(dims, 3)}, centroid={np.round(centroid, 3)}")
        kept += 1

    print(f"Wrote {kept} '{target_class}' mesh(es)")

manifest_path = os.path.join(SEG_DIR, "objects_3d.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n{'='*50}")
print(f"Total objects written: {len(manifest)}")
print(f"Meshes: {MESH_DIR}")
print(f"Manifest: {manifest_path}")
print("\nIf a class got 0-1 clusters but you expected more, lower --dbscan-eps.")
print("If objects came out fragmented/hollow, try a smaller --alpha, or lower --min-frame-votes.")