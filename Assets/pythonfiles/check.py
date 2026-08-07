"""
check.py

Diagnostic / tuning helper. Run this whenever detect_objects.py's
clustering looks wrong (too many tiny clusters, or everything merged
into one blob) to find a better DBSCAN_EPS before editing the real script.
"""
import json
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", default=r"C:\Users\Arni\UnityProjects\TestCapture",
                     help="MUST match whatever run_pipeline.py used for this capture.")
parser.add_argument("--max-steps", type=int, default=7000,
                     help="Must match run_pipeline.py's MAX_STEPS.")
args = parser.parse_args()
BASE_DIR = args.base_dir
print(f"Using BASE_DIR: {BASE_DIR}")

with open(rf"{BASE_DIR}\segmentation\detections_seg.json") as f:
    d = json.load(f)
assert len(d) > 0, "detections_seg.json is empty -- run run_yolo_segmentation.py first"

for img, dets in d.items():
    if dets:
        print(f"Sample detections — image: {img}")
        for det in dets[:3]:
            print(f'  {det["class"]} conf={det["confidence"]:.2f} mask_key={det["mask_key"]}')
        break

print("\nPLY / scene scale info:")
ply = PlyData.read(rf"{BASE_DIR}\output\ply\point_cloud_{args.max_steps - 1}.ply")
v = ply["vertex"]
xyz = np.stack([v["x"].astype(np.float32), v["y"].astype(np.float32), v["z"].astype(np.float32)], axis=1)
assert len(xyz) > 0, "Trained .ply has zero points -- training likely failed"
print(f"PLY bounds: min={xyz.min(axis=0).round(3)}, max={xyz.max(axis=0).round(3)}")
print(f"Scene extent: {(xyz.max(axis=0) - xyz.min(axis=0)).round(3)}")
print("(gsplat's normalize step scales the scene so median camera distance")
print(" from center is ~1.0 -- if these numbers are wildly different from")
print(" that, normalization.json may be stale, re-run compute_transform.py)")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd.voxel_down_sample(voxel_size=0.01)
xyz_c = np.asarray(pcd.points)
print(f"\nAfter cleanup/downsample: {len(xyz_c)} points")

print("\nSweeping eps values (normalized-scene scale, so much smaller than before):")
for eps in [0.02, 0.03, 0.05, 0.08, 0.12, 0.2]:
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=20, print_progress=False))
    n = labels.max() + 1
    sizes = [int((labels == i).sum()) for i in range(n)]
    print(f"eps={eps}: {n} clusters, sizes={sorted(sizes, reverse=True)[:8]}")