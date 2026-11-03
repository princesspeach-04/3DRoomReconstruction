"""
replace_splats.py

Step (c) of the gen-AI replacement: removes the ORIGINAL (blurry/distorted) splats
inside every replaced object's region, so the TripoSR mesh replaces it instead of
overlapping it.

Reads  output/ply/point_cloud_<N><ply-suffix>.ply   (default: the _clean one)
       segmentation/objects_3d.json                   (objects with mesh_source=generative_replace_triposr
                                                       and a replace_region written by generate_replacement.py)
Writes output/ply/point_cloud_<N><ply-suffix><out-suffix>.ply   (default: ..._clean_replaced.ply)
       and "splats_removed" into each object in objects_3d.json.
The input PLY is never modified.

The region is the replacement mesh's bounding box in PLY space, grown sideways and
upwards by a margin and NOT downwards (so the table/floor under the object stays).
"Up" comes from the world_up vector stored by generate_replacement.py.

Afterwards: recreate the Gaussian Splat asset in Unity from the new PLY, and (optional)
rebuild the collider:  build_collider_voxel.py --capture-dir <dir> --ply-suffix _clean_replaced

Usage:
  python replace_splats.py --base-dir C:\\...\\video_Capture --max-steps 30000
  python replace_splats.py --base-dir ... --ids laptop_0 --margin-side 0.2
"""
import os
import json
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

ap = argparse.ArgumentParser()
ap.add_argument("--base-dir", required=True)
ap.add_argument("--max-steps", type=int, default=30000)
ap.add_argument("--ply-suffix", default="_clean")
ap.add_argument("--out-suffix", default="_replaced")
ap.add_argument("--ids", nargs="+", default=None, help="Only these object ids (default: all replaced objects).")
ap.add_argument("--margin-side", type=float, default=0.12, help="fraction of object size added on the sides")
ap.add_argument("--margin-up", type=float, default=0.10, help="fraction of object size added above")
ap.add_argument("--margin-down", type=float, default=0.0, help="fraction of object size added below (keeps the table/floor)")
args = ap.parse_args()

base = args.base_dir
ply_dir = os.path.join(base, "output", "ply")
in_path = os.path.join(ply_dir, f"point_cloud_{args.max_steps - 1}{args.ply_suffix}.ply")
out_path = os.path.join(ply_dir, f"point_cloud_{args.max_steps - 1}{args.ply_suffix}{args.out_suffix}.ply")
manifest_path = os.path.join(base, "segmentation", "objects_3d.json")
assert os.path.exists(in_path), f"PLY not found: {in_path}"

with open(manifest_path) as f:
    manifest = json.load(f)
objs = [o for o in manifest if o.get("mesh_source") == "generative_replace_triposr" and "replace_region" in o]
if args.ids:
    objs = [o for o in objs if o["id"] in set(args.ids)]
print(f"{len(objs)} replaced object(s) to cut out")
if not objs:
    raise SystemExit("Nothing to do (run generate_replacement.py first).")

print(f"Loading {in_path} ...")
ply = PlyData.read(in_path)
data = ply["vertex"].data
xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
keep = np.ones(len(xyz), dtype=bool)

for o in objs:
    lo = np.array(o["replace_region"]["min"], float)
    hi = np.array(o["replace_region"]["max"], float)
    size = hi - lo
    up = np.array(o.get("world_up", [0, 0, 1]), float)
    ax = int(np.argmax(np.abs(up)))
    sign = 1.0 if up[ax] >= 0 else -1.0
    lo = lo - args.margin_side * size
    hi = hi + args.margin_side * size
    below, above = args.margin_down * size[ax], args.margin_up * size[ax]
    # undo the symmetric side margin on the up axis, then apply up/down margins
    lo[ax] = o["replace_region"]["min"][ax]
    hi[ax] = o["replace_region"]["max"][ax]
    if sign > 0:
        lo[ax] -= below
        hi[ax] += above
    else:
        hi[ax] += below
        lo[ax] -= above
    inside = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    n_new = int((inside & keep).sum())
    keep &= ~inside
    o["splats_removed"] = n_new
    print(f"  {o['id']}: removed {n_new} splats (region {np.round(lo, 3)} .. {np.round(hi, 3)})")

n_kept = int(keep.sum())
print(f"\nKeeping {n_kept}/{len(xyz)} splats")
assert n_kept > 500, "Almost everything removed - check replace_region values"

PlyData([PlyElement.describe(data[keep], "vertex")], text=False).write(out_path)
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote {out_path}")
print("Now point Unity's Gaussian Splat Creator at this file and recreate the asset.")