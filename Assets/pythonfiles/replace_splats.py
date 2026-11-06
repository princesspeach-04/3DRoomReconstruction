"""
replace_splats.py   (v2: removes the old splats AND fills the hole they leave)

Step (c) of the gen-AI replacement.

1. REMOVE: cuts the ORIGINAL (blurry/distorted) splats inside every replaced object's region, so the
   TripoSR mesh replaces it instead of overlapping it. (No cut below the object along world-up, so the
   support surface stays.)
2. FILL: whatever was behind / under the object was never seen by any camera, so removing the object
   leaves a hole (you see black / the void). The fill step makes that area look complete:
     - takes the splats in a ring around the hole,
     - finds the dominant planar surfaces there (RANSAC): a table top, the floor, a chair seat, a
       chair back, a wall ... up to 3,
     - keeps the ones that pass through the hole,
     - extends each plane across the hole with NEW splats whose colour / SH / opacity / scale / rotation
       are copied from nearby real splats of that same surface (so wood grain, cushion colour, etc.
       continue instead of a flat patch). Only empty places get filled; nothing is added where real
       splats already exist.
   Works for any object / any surface (class-agnostic).

Reads  output/ply/point_cloud_<N><ply-suffix>.ply   (default: the _clean one)
       segmentation/objects_3d.json  (objects with mesh_source=generative_replace_triposr + replace_region)
Writes output/ply/point_cloud_<N><ply-suffix><out-suffix>.ply   (default: ..._clean_replaced.ply)
       and "splats_removed" / "fill_splats_added" per object in objects_3d.json.
The input PLY is never modified.

Afterwards: recreate the Gaussian Splat asset in Unity from the new PLY. The collider rebuild
(build_collider_voxel.py --ply-suffix _clean_replaced) will include the filled surface.

Usage:
  python replace_splats.py --base-dir C:\\...\\video_Capture --max-steps 30000
  python replace_splats.py --base-dir ... --fill-debug      (filled splats shown in cyan, to see what was added)
  python replace_splats.py --base-dir ... --no-fill         (removal only)
"""
import os
import json
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--base-dir", required=True)
ap.add_argument("--max-steps", type=int, default=30000)
ap.add_argument("--ply-suffix", default="_clean")
ap.add_argument("--out-suffix", default="_replaced")
ap.add_argument("--ids", nargs="+", default=None, help="Only these object ids (default: all replaced objects).")
ap.add_argument("--margin-side", type=float, default=0.12, help="fraction of object size added on the sides")
ap.add_argument("--margin-up", type=float, default=0.10, help="fraction of object size added above")
ap.add_argument("--margin-down", type=float, default=0.0, help="fraction of object size added below (keeps the table/floor)")
ap.add_argument("--no-fill", action="store_true", help="only remove, do not fill the hole")
ap.add_argument("--fill-debug", action="store_true", help="colour the filled splats cyan so you can see them")
ap.add_argument("--fill-margin", type=float, default=0.15,
                help="how far (fraction of object size) beyond the removed box the fill may extend")
ap.add_argument("--ring", type=float, default=0.6,
                help="width of the neighbourhood (fraction of object size) used to find surfaces")
ap.add_argument("--max-planes", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
base = args.base_dir
ply_dir = os.path.join(base, "output", "ply")
in_path = os.path.join(ply_dir, f"point_cloud_{args.max_steps - 1}{args.ply_suffix}.ply")
out_path = os.path.join(ply_dir, f"point_cloud_{args.max_steps - 1}{args.ply_suffix}{args.out_suffix}.ply")
manifest_path = os.path.join(base, "segmentation", "objects_3d.json")
assert os.path.exists(in_path), f"PLY not found: {in_path}"


# ----------------------------------------------------------------------------
# hole filling
# ----------------------------------------------------------------------------
def ransac_planes(P, rng, tol, max_planes, iters=300, min_inliers=80, min_frac=0.05):
    """Dominant planes of a point set. Returns [(centre, unit normal, inlier index array)]."""
    planes = []
    remaining = np.arange(len(P))
    for _ in range(max_planes):
        if len(remaining) < min_inliers:
            break
        pts = P[remaining]
        sub = pts if len(pts) <= 6000 else pts[rng.choice(len(pts), 6000, replace=False)]
        best = None
        for _it in range(iters):
            idx = rng.choice(len(sub), 3, replace=False)
            a, b, c = sub[idx]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            cnt = int((np.abs((sub - a) @ n) < tol).sum())
            if best is None or cnt > best[0]:
                best = (cnt, n, a)
        if best is None:
            break
        _, n, a = best
        for _refit in range(2):                      # refine with SVD on the inliers
            inl = np.abs((pts - a) @ n) < tol
            if inl.sum() < 10:
                break
            Q = pts[inl]
            a = Q.mean(0)
            n = np.linalg.svd(Q - a, full_matrices=False)[2][2]
        all_in = np.where(np.abs((P - a) @ n) < tol)[0]
        if len(all_in) < max(min_inliers, min_frac * len(P)):
            break
        planes.append((a, n, all_in))
        remaining = np.setdiff1d(remaining, all_in)
    return planes


def fill_region(data_k, xyz_k, tree_k, lo, hi, a):
    """New splats extending the surfaces around the removed box [lo, hi] across it."""
    size = hi - lo
    smax = float(size.max())
    ring_lo, ring_hi = lo - a.ring * smax, hi + a.ring * smax
    idx = np.where(np.all((xyz_k >= ring_lo) & (xyz_k <= ring_hi), axis=1))[0]
    if len(idx) < 150:
        return [], 0
    P = xyz_k[idx]
    tol = max(0.004, 0.03 * smax)
    planes = ransac_planes(P, rng, tol, a.max_planes)
    center = (lo + hi) / 2
    diag = float(np.linalg.norm(size))
    exp_lo, exp_hi = lo - a.fill_margin * size, hi + a.fill_margin * size
    corners = np.array([[x, y, z] for x in (exp_lo[0], exp_hi[0]) for y in (exp_lo[1], exp_hi[1])
                        for z in (exp_lo[2], exp_hi[2])])
    out, n_planes = [], 0
    for (c, n, inl) in planes:
        if abs((center - c) @ n) > 0.5 * diag:       # plane does not pass through / next to the hole
            continue
        Q = P[inl]
        tq = cKDTree(Q)
        d2 = tq.query(Q, k=2)[0][:, 1]
        s = float(np.clip(np.median(d2), 0.0015, 0.1 * smax))
        ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(n, ref); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        rel = corners - c
        ua, ub = rel @ u, rel @ v
        A, B = np.meshgrid(np.arange(ua.min(), ua.max(), s), np.arange(ub.min(), ub.max(), s))
        A = A.ravel() + rng.uniform(-0.35 * s, 0.35 * s, A.size)
        B = B.ravel() + rng.uniform(-0.35 * s, 0.35 * s, B.size)
        pts = c + A[:, None] * u + B[:, None] * v
        keep = np.all((pts >= exp_lo) & (pts <= exp_hi), axis=1)
        pts = pts[keep]
        if len(pts) == 0:
            continue
        keep = tree_k.query(pts)[0] > 0.7 * s              # only empty places
        pts = pts[keep]
        if len(pts) == 0:
            continue
        dq = tq.query(pts)[0]
        pts = pts[dq <= 1.2 * smax]                        # don't extend a plane far from its evidence
        if len(pts) == 0:
            continue
        if len(pts) > 60000:
            pts = pts[rng.choice(len(pts), 60000, replace=False)]
        k = min(6, len(Q))
        dd, ii = tq.query(pts, k=k)
        dd = dd.reshape(len(pts), -1)
        ii = ii.reshape(len(pts), -1)
        w = 1.0 / (dd + 1e-3 * smax)
        cw = np.cumsum(w / w.sum(1, keepdims=True), axis=1)
        pick = (rng.random(len(pts))[:, None] > cw).sum(1).clip(0, k - 1)
        src = ii[np.arange(len(pts)), pick]
        new = data_k[idx[inl[src]]].copy()
        new["x"], new["y"], new["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        names = new.dtype.names
        if "nx" in names:
            new["nx"], new["ny"], new["nz"] = n[0], n[1], n[2]
        if a.fill_debug and "f_dc_0" in names:
            for nm, val in zip(("f_dc_0", "f_dc_1", "f_dc_2"), (-1.772, 1.772, 1.772)):   # cyan
                new[nm] = val
            for nm in names:
                if nm.startswith("f_rest_"):
                    new[nm] = 0.0
        out.append(new)
        n_planes += 1
    return out, n_planes


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
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

boxes = []
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
    boxes.append((o, lo, hi))
    print(f"  {o['id']}: removed {n_new} splats (region {np.round(lo, 3)} .. {np.round(hi, 3)})")

n_kept = int(keep.sum())
print(f"\nKeeping {n_kept}/{len(xyz)} splats")
assert n_kept > 500, "Almost everything removed - check replace_region values"

data_k = data[keep]
parts = [data_k]
if not args.no_fill:
    xyz_k = xyz[keep]
    tree_k = cKDTree(xyz_k)
    total = 0
    print("\nFilling the holes (extending the surrounding surfaces across them) ...")
    for o, lo, hi in boxes:
        new_list, n_planes = fill_region(data_k, xyz_k, tree_k, lo, hi, args)
        n_add = int(sum(len(x) for x in new_list))
        o["fill_splats_added"] = n_add
        o["fill_planes"] = n_planes
        total += n_add
        if n_add:
            parts += new_list
            print(f"  {o['id']}: +{n_add} splats on {n_planes} surface(s)")
        else:
            print(f"  {o['id']}: nothing to fill (no surface found around it, or the hole is empty space)")
    print(f"Added {total} splats in total" + ("  (cyan = filled)" if args.fill_debug else ""))

final = np.concatenate(parts) if len(parts) > 1 else data_k
PlyData([PlyElement.describe(final, "vertex")], text=False).write(out_path)
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote {out_path}  ({len(final)} splats)")
print("Now point Unity's Gaussian Splat Creator at this file and recreate the asset.")