"""
build_room_mesh.py

Reconstructs the WHOLE room as a single mesh, for COLLISION ONLY -- this
is the "hybrid" approach: the splat keeps rendering the visuals in
Unity exactly as before, this mesh is an invisible collider so a player
can walk around with real geometry instead of RoomColliderBuilder's 6
hardcoded boxes.

Uses Poisson surface reconstruction, NOT the alpha-shape method
build_object_meshes.py uses for chairs -- alpha-shapes are better for
isolated objects with real gaps (chair legs), Poisson is better for
room-scale continuous surfaces (floor/walls/ceiling) since it produces
a smooth, closed surface instead of following every point-cloud gap
literally (which would leave the room full of holes wherever the splat
itself has noisy/sparse coverage).

Usage:
    python build_room_mesh.py --base-dir <capture_dir> --max-steps 30000
"""
import os
import argparse
import numpy as np
import open3d as o3d
from plyfile import PlyData

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", required=True)
parser.add_argument("--max-steps", type=int, default=30000,
                     help="Must match whatever step count you actually trained to.")
parser.add_argument("--voxel-size", type=float, default=0.01,
                     help="Downsample voxel size before reconstruction, in the "
                          "normalized scene scale (median camera distance ~1.0 "
                          "-- NOT real meters). Smaller = more detail but slower "
                          "and noisier; check.py's PLY bounds print gives you a "
                          "sense of scene scale if you need to tune this.")
parser.add_argument("--poisson-depth", type=int, default=9,
                     help="Higher = more surface detail but more memory/time and "
                          "more prone to hallucinating surface where there's no "
                          "real data. 8-10 is the usual room-scale range.")
parser.add_argument("--density-quantile", type=float, default=0.02,
                     help="Poisson reconstruction extrapolates a smooth surface "
                          "PAST the actual data extent (a well-known artifact -- "
                          "it wants to close the mesh into a blob). This crops "
                          "away the lowest-density fraction of vertices, which "
                          "are almost always that extrapolated fringe, not real "
                          "geometry. Raise this if you still see a rounded shell "
                          "around the real room; lower it if real walls are "
                          "getting cut away.")
parser.add_argument("--simplify-to", type=int, default=0,
                     help="Optional target triangle count via quadric decimation "
                          "(e.g. 50000). A non-convex MeshCollider has no hard "
                          "triangle limit, but very high counts hurt raycast/"
                          "physics performance at runtime. 0 = don't simplify.")
args = parser.parse_args()

BASE_DIR = args.base_dir
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
print(f"Using BASE_DIR: {BASE_DIR}")

ply_path = os.path.join(OUTPUT_DIR, "ply", f"point_cloud_{args.max_steps - 1}.ply")
assert os.path.exists(ply_path), f"Trained PLY not found at {ply_path}"
ply = PlyData.read(ply_path)
v = ply["vertex"]
points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
print(f"Loaded {len(points)} raw splat points")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

print("Removing statistical outliers...")
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

print(f"Downsampling at voxel size {args.voxel_size}...")
pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
print(f"{len(pcd.points)} points after cleanup/downsample")

print("Estimating normals (consistent tangent-plane orientation, since "
      "there's no per-point camera-facing info at this stage)...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=args.voxel_size * 4, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(k=15)

print(f"Running Poisson reconstruction (depth={args.poisson_depth})...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=args.poisson_depth)
densities = np.asarray(densities)

print(f"Cropping low-density fringe (bottom {args.density_quantile*100:.0f}%)...")
threshold = np.quantile(densities, args.density_quantile)
mesh.remove_vertices_by_mask(densities < threshold)

print("Cleaning mesh...")
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
mesh.remove_non_manifold_edges()  # important for a stable runtime MeshCollider

if args.simplify_to > 0 and len(mesh.triangles) > args.simplify_to:
    print(f"Simplifying {len(mesh.triangles)} -> ~{args.simplify_to} triangles...")
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.simplify_to)

mesh.compute_vertex_normals()

assert len(mesh.triangles) > 0, (
    "Reconstruction produced zero triangles -- try a lower --density-quantile, "
    "or check the point cloud isn't empty after outlier removal."
)

out_path = os.path.join(OUTPUT_DIR, "room_mesh.obj")
o3d.io.write_triangle_mesh(out_path, mesh)

bbox = mesh.get_axis_aligned_bounding_box()
print(f"\nRoom mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
print(f"Bounds: min={bbox.get_min_bound().round(3)}, max={bbox.get_max_bound().round(3)}")
print(f"Wrote: {out_path}")
print("\nThis mesh is for COLLISION ONLY in Unity -- the splat still renders the")
print("visuals. If it looks too smooth/rounded (Poisson's known failure mode) or")
print("real walls got cropped away, adjust --density-quantile and re-run.")