"""
build_object_meshes.py

Multi-class object mesh extraction. For every class present in
detections_seg.json (or every class in --target-classes if given):

  1. Projects every trained Gaussian-Splat point into every camera frame.
  2. Checks whether the projected pixel falls inside a YOLO-SEG mask for
     that class in that frame.
  3. A point that lands inside a mask in enough DIFFERENT frames gets a
     vote for that class.
  4. DBSCAN-clusters voted points into individual object instances.
  5. Reconstructs a real mesh per cluster with Open3D alpha-shape.
  6. Records quality metrics (triangle_count, bbox_aspect_ratio) per
     object into the manifest, for a later treatment-classification step
     to decide keep/regularize/generative-replace without needing an AI
     call.

Requires (already produced earlier in your pipeline):
  - output/ply/point_cloud_<N>.ply           (trained splat, from gsplat)
  - sparse/0/images.bin, cameras.bin          (raw COLMAP poses/intrinsics)
  - output/normalization.json                 (from compute_transform.py)
  - segmentation/detections_seg.json + masks/ (from run_yolo_segmentation.py)

Usage:
    python build_object_meshes.py --base-dir <capture_dir> --max-steps 30000
    python build_object_meshes.py --base-dir <capture_dir> --max-steps 30000 --target-classes chair table

NOTE: reads point_cloud_<N>_clean.ply by default (run clean_gaussians.py first).
"""
import os
import json
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData

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
                     help="A point must land inside the mask in at least "
                          "this many distinct frames to count as 'on the "
                          "object'. Raise this if you get floor/wall bleed; "
                          "lower it if objects come out too sparse/holey.")
parser.add_argument("--dbscan-eps", type=float, default=0.05,
                     help="Same normalized-scene scale as check.py's sweep "
                          "-- run check.py first if unsure.")
parser.add_argument("--dbscan-min-points", type=int, default=30)
parser.add_argument("--alpha", type=float, default=0.05,
                     help="Open3D alpha-shape parameter -- smaller = tighter "
                          "surface that follows concave detail, larger = "
                          "smoother but may fill gaps.")
parser.add_argument("--min-cluster-size", type=int, default=80,
                     help="Discard DBSCAN clusters smaller than this -- "
                          "filters out noise blobs that aren't real objects.")
args = parser.parse_args()

BASE_DIR = args.base_dir
SPARSE_DIR = os.path.join(BASE_DIR, "sparse", "0")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MESH_DIR = os.path.join(OUTPUT_DIR, "meshes")
SEG_DIR = os.path.join(BASE_DIR, "segmentation")
MASKS_DIR = os.path.join(SEG_DIR, "masks")
print(f"Using BASE_DIR: {BASE_DIR}")
os.makedirs(MESH_DIR, exist_ok=True)

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
print(f"Loaded {len(points)} trained splat points")

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

manifest = []
pts_h_all = np.hstack([points, np.ones((len(points), 1))])

for target_class in target_classes:
    print(f"\n{'='*50}")
    print(f"Class: {target_class}")
    print(f"{'='*50}")

    vote_counts = np.zeros(len(points), dtype=np.int32)
    frames_with_target = [f for f, ents in detections.items()
                           if any(e["class"] == target_class for e in ents)]
    print(f"{len(frames_with_target)} frames contain '{target_class}' detections")
    if len(frames_with_target) == 0:
        continue

    for fname in frames_with_target:
        if fname not in frame_c2w:
            continue
        c2w = frame_c2w[fname]
        w2c = np.linalg.inv(c2w)

        cam_pts = (w2c @ pts_h_all.T).T[:, :3]

        stem = os.path.splitext(fname)[0]
        npz_path = os.path.join(MASKS_DIR, f"{stem}.npz")
        if not os.path.exists(npz_path):
            continue
        masks = np.load(npz_path)

        for ent in detections[fname]:
            if ent["class"] != target_class:
                continue
            mask = masks[ent["mask_key"]]
            ui, vi, ok = project(cam_pts, frame_cam[fname], mask.shape)
            hit = ok & mask[vi, ui].astype(bool)
            vote_counts[hit] += 1

    voted = points[vote_counts >= args.min_frame_votes]
    print(f"{len(voted)} points reached >= {args.min_frame_votes} frame votes for '{target_class}'")
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

        # ---- quality metrics for later treatment classification ----
        sorted_dims = sorted(dims, reverse=True)
        bbox_aspect_ratio = sorted_dims[0] / max(sorted_dims[-1], 1e-6)
        triangle_count = len(mesh.triangles)
        point_count = len(cluster_pts)

        obj_name = f"{target_class}_{kept}.obj"
        obj_path = os.path.join(MESH_DIR, obj_name)
        mesh.triangle_normals = o3d.utility.Vector3dVector()  # OBJ cannot store them
        o3d.io.write_triangle_mesh(obj_path, mesh)

        avg_confidence = float(np.mean([
            e["confidence"] for f in frames_with_target for e in detections[f]
            if e["class"] == target_class
        ]))

                # ---- find the best-matching frame for THIS SPECIFIC instance,
        # not just "any frame with this class" -- needed so generative
        # replacement can crop the right object when multiple instances
        # of the same class exist (e.g. 3 different chairs). Reuses the
        # cluster's own (uncentered) points and checks, per frame, how
        # many actually project inside a real mask for this class. ----
        best_frame, best_mask_key, best_overlap, best_conf = None, None, -1, 0.0
        cluster_pts_world = cluster_pts  # original coords, pre-centering
        cpts_h = np.hstack([cluster_pts_world, np.ones((len(cluster_pts_world), 1))])

        for fname in frames_with_target:
            if fname not in frame_c2w:
                continue
            w2c_f = np.linalg.inv(frame_c2w[fname])
            cam_pts_f = (w2c_f @ cpts_h.T).T[:, :3]

            stem_f = os.path.splitext(fname)[0]
            npz_path_f = os.path.join(MASKS_DIR, f"{stem_f}.npz")
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
            "source_ply": os.path.basename(ply_path),
        })
        print(f"  cluster {cluster_id} -> {obj_name}: "
              f"{point_count} pts, {triangle_count} triangles, "
              f"aspect={bbox_aspect_ratio:.2f}, centroid={np.round(centroid, 3)}")
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