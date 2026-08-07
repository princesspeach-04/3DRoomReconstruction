"""
build_object_meshes.py

Supersedes your old detect_objects.py's approach. Instead of
backprojecting bounding boxes (which gives you a rectangular blob no
matter what's really in the box), this:

  1. Projects every trained Gaussian-Splat point into every camera frame
     using that frame's pose.
  2. Checks whether the projected pixel falls inside a YOLO-SEG mask for
     the target class (e.g. "chair") in that frame.
  3. A point that lands inside a mask in enough DIFFERENT frames gets a
     "chair" vote. This is what makes it pixel-accurate instead of
     box-accurate -- a point on the floor next to a chair's bounding box
     will land inside the box but almost never inside the actual mask
     silhouette across multiple viewpoints.
  4. Runs DBSCAN on the voted points to split them into separate physical
     chairs (multiple chairs in a scene become multiple clusters
     naturally, even though we never track per-frame instance identity
     across frames).
  5. Reconstructs an actual surface mesh per cluster with Open3D
     (alpha-shape), instead of just a point blob or a box -- this mesh
     IS the collider shape you'll import into Unity.

Requires (already produced earlier in your pipeline):
  - output/ply/point_cloud_<N>.ply           (trained splat, from gsplat)
  - sparse/0/images.bin, cameras.bin          (raw COLMAP poses/intrinsics)
  - output/normalization.json                 (from compute_transform.py)
  - segmentation/detections_seg.json + masks/ (from run_yolo_segmentation.py)

Usage:
    python build_object_meshes.py --base-dir <capture_dir> --max-steps 7000
"""
import os
import json
import struct
import argparse
import collections
import numpy as np
import open3d as o3d
from plyfile import PlyData

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", required=True)
parser.add_argument("--max-steps", type=int, default=7000,
                     help="Must match run_pipeline.py's MAX_STEPS -- used to "
                          "find point_cloud_<max_steps-1>.ply")
parser.add_argument("--target-class", default="chair")
parser.add_argument("--min-frame-votes", type=int, default=3,
                     help="A point must land inside the mask in at least "
                          "this many distinct frames to count as 'on the "
                          "object'. Raise this if you get floor/wall bleed; "
                          "lower it if chairs come out too sparse/holey.")
parser.add_argument("--dbscan-eps", type=float, default=0.05,
                     help="Same normalized-scene scale as check.py's sweep "
                          "-- run check.py first if unsure.")
parser.add_argument("--dbscan-min-points", type=int, default=30)
parser.add_argument("--alpha", type=float, default=0.05,
                     help="Open3D alpha-shape parameter -- smaller = tighter "
                          "surface that follows concave detail (gaps between "
                          "chair legs), larger = smoother but may fill gaps.")
parser.add_argument("--min-cluster-size", type=int, default=80,
                     help="Discard DBSCAN clusters smaller than this -- "
                          "filters out noise blobs that aren't real chairs.")
args = parser.parse_args()

BASE_DIR = args.base_dir
SPARSE_DIR = os.path.join(BASE_DIR, "sparse", "0")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MESH_DIR = os.path.join(OUTPUT_DIR, "meshes")
SEG_DIR = os.path.join(BASE_DIR, "segmentation")
MASKS_DIR = os.path.join(SEG_DIR, "masks")
print(f"Using BASE_DIR: {BASE_DIR}")
os.makedirs(MESH_DIR, exist_ok=True)

# ---- COLMAP binary readers (same approach as compute_transform.py) ----
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "name"])
Camera = collections.namedtuple("Camera", ["fx", "fy", "cx", "cy", "width", "height"])


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            struct.unpack("<i", f.read(4))  # camera_id
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(16 * n_pts)
            f.read(8 * n_pts)
            images[img_id] = Image(img_id, qvec, tvec, name)
    return images


def read_cameras_binary(path):
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        cam_id = struct.unpack("<i", f.read(4))[0]
        model_id = struct.unpack("<i", f.read(4))[0]
        width = struct.unpack("<Q", f.read(8))[0]
        height = struct.unpack("<Q", f.read(8))[0]
        # PINHOLE (model_id 1): fx, fy, cx, cy
        fx, fy, cx, cy = struct.unpack("<4d", f.read(32))
    return Camera(fx, fy, cx, cy, width, height)


def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ])


# ---- load everything ----

ply_path = os.path.join(OUTPUT_DIR, "ply", f"point_cloud_{args.max_steps - 1}.ply")
assert os.path.exists(ply_path), f"Trained PLY not found at {ply_path}"
ply = PlyData.read(ply_path)
v = ply["vertex"]
points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
print(f"Loaded {len(points)} trained splat points")

images = read_images_binary(os.path.join(SPARSE_DIR, "images.bin"))
camera = read_cameras_binary(os.path.join(SPARSE_DIR, "cameras.bin"))
print(f"Loaded {len(images)} camera poses, intrinsics fx={camera.fx:.1f} fy={camera.fy:.1f}")

with open(os.path.join(OUTPUT_DIR, "normalization.json")) as f:
    T = np.array(json.load(f)["transform"])

with open(os.path.join(SEG_DIR, "detections_seg.json")) as f:
    detections = json.load(f)

# ---- build normalized camera-to-world matrices (raw COLMAP -> ply space) ----
frame_c2w = {}
for img in images.values():
    R = qvec_to_rotmat(img.qvec)
    t = np.array(img.tvec)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)
    # apply the SAME transform compute_transform.py applied to the point
    # cloud/cameras before training -- this is the step that keeps this
    # script in the same coordinate space as the trained .ply. Skipping
    # it is exactly the bug class compute_transform.py exists to prevent.
    c2w_t = T @ c2w
    scale = np.linalg.norm(c2w_t[:3, 0])
    c2w_t[:3, :3] /= scale
    frame_c2w[img.name] = c2w_t

# ---- vote pass: project every point into every frame with a mask ----
vote_counts = np.zeros(len(points), dtype=np.int32)
frames_with_target = [f for f, ents in detections.items()
                       if any(e["class"] == args.target_class for e in ents)]
print(f"\n{len(frames_with_target)} frames contain '{args.target_class}' detections")

for fname in frames_with_target:
    if fname not in frame_c2w:
        continue
    c2w = frame_c2w[fname]
    w2c = np.linalg.inv(c2w)

    pts_h = np.hstack([points, np.ones((len(points), 1))])
    cam_pts = (w2c @ pts_h.T).T[:, :3]
    in_front = cam_pts[:, 2] > 1e-6

    u = camera.fx * cam_pts[:, 0] / np.clip(cam_pts[:, 2], 1e-6, None) + camera.cx
    v_ = camera.fy * cam_pts[:, 1] / np.clip(cam_pts[:, 2], 1e-6, None) + camera.cy
    in_bounds = in_front & (u >= 0) & (u < camera.width) & (v_ >= 0) & (v_ < camera.height)

    stem = os.path.splitext(fname)[0]
    npz_path = os.path.join(MASKS_DIR, f"{stem}.npz")
    if not os.path.exists(npz_path):
        continue
    masks = np.load(npz_path)

    ui = u[in_bounds].astype(np.int32)
    vi = v_[in_bounds].astype(np.int32)
    idx = np.where(in_bounds)[0]

    for ent in detections[fname]:
        if ent["class"] != args.target_class:
            continue
        mask = masks[ent["mask_key"]]
        hit = mask[vi, ui]
        vote_counts[idx[hit]] += 1

voted = points[vote_counts >= args.min_frame_votes]
print(f"\n{len(voted)} points reached >= {args.min_frame_votes} frame votes for '{args.target_class}'")
assert len(voted) >= args.dbscan_min_points, (
    "Too few voted points to cluster -- lower --min-frame-votes, check "
    "mask quality, or verify detections_seg.json actually has this class."
)

# ---- cluster into individual chairs ----
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(voted)
labels = np.array(pcd.cluster_dbscan(
    eps=args.dbscan_eps, min_points=args.dbscan_min_points, print_progress=False))
n_clusters = labels.max() + 1
print(f"DBSCAN found {n_clusters} raw clusters")

manifest = []
kept = 0
for cluster_id in range(n_clusters):
    cluster_pts = voted[labels == cluster_id]
    if len(cluster_pts) < args.min_cluster_size:
        continue

    # Center the mesh on its own centroid BEFORE reconstruction, so the
    # exported .obj is object-local (like a normal prefab) instead of
    # carrying absolute scene coordinates baked into every vertex. Your
    # existing SceneObjectSpawner.cs already expects this split -- it
    # applies "position" via transform.localPosition separately from the
    # mesh geometry -- so this keeps mesh_file compatible with that,
    # instead of introducing a second, conflicting way of encoding position.
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

    bbox = cluster_pcd.get_axis_aligned_bounding_box()  # now local-space, centroid already subtracted
    dims = (bbox.get_max_bound() - bbox.get_min_bound()).tolist()

    obj_name = f"{args.target_class}_{kept}.obj"
    obj_path = os.path.join(MESH_DIR, obj_name)
    o3d.io.write_triangle_mesh(obj_path, mesh)

    # avg_confidence is a GLOBAL average across every frame that
    # contributed a "chair" detection, not a true per-instance score --
    # the voting pass never tracks which frame's mask belongs to which
    # physical chair, so a real per-cluster confidence isn't available.
    # Good enough for classFilter/minConfidence gating; don't over-trust
    # small differences between two clusters' confidence values.
    avg_confidence = float(np.mean([
        e["confidence"] for f in frames_with_target for e in detections[f]
        if e["class"] == args.target_class
    ]))

    manifest.append({
        "id": f"{args.target_class}_{kept}",
        "class": args.target_class,
        "position": centroid.tolist(),
        "size": dims,
        "point_count": len(cluster_pts),
        "confidence": avg_confidence,
        "mass": 5.0,
        "drag": 0.5,
        "mesh_file": obj_path,
    })
    print(f"  cluster {cluster_id} -> {obj_name}: "
          f"{len(cluster_pts)} pts, {len(mesh.triangles)} triangles, "
          f"centroid={np.round(centroid, 3)}")
    kept += 1

manifest_path = os.path.join(SEG_DIR, "objects_3d.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nWrote {kept} object mesh(es) to {MESH_DIR}")
print(f"Manifest: {manifest_path}")
print("\nIf you got 0-1 clusters but expected more chairs, lower --dbscan-eps.")
print("If chairs came out fragmented/hollow, try a smaller --alpha, or lower --min-frame-votes.")