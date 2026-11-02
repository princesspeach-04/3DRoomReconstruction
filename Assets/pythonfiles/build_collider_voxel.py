"""
build_collider_voxel.py
Voxel-density collision mesh from a cleaned Gaussian-splat PLY.
Replaces Poisson (no normals needed, no membrane over empty space, floaters removed by blob filter).

Pipeline: opacity filter -> FIND AND REMOVE THE FLOOR (so chairs/table are no longer glued
together by floor splats) -> density grid -> smooth -> threshold -> keep each separate blob
(= each object) -> marching cubes -> quadric simplify -> outward winding
-> OBJ with one group per part + a clean flat floor slab (same coordinate space as the PLY).
Also writes collider_parts.json (one entry per part) next to the OBJ.

Usage:
  python build_collider_voxel.py --capture-dir C:\\...\\video_Capture
  python build_collider_voxel.py --ply path\\point_cloud_30000_clean.ply --out room_mesh.obj
"""
import argparse, glob, json, os, re, sys
import numpy as np
from plyfile import PlyData
from scipy import ndimage
from skimage import measure


def find_ply(capture_dir, suffix):
    pats = glob.glob(os.path.join(capture_dir, "**", f"point_cloud_*{suffix}.ply"), recursive=True)
    if not pats:
        pats = [p for p in glob.glob(os.path.join(capture_dir, "**", "point_cloud_*.ply"), recursive=True)
                if "_clean" not in os.path.basename(p)]
    if not pats:
        sys.exit("No point_cloud_*.ply found")
    def step(p):
        m = re.search(r"point_cloud_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(pats, key=step)


def load_splats(path, min_opacity):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
    keep = op >= min_opacity
    sc = None
    if "scale_0" in v.data.dtype.names:
        sc = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64))
        sc = sc[keep]
    return xyz[keep], op[keep], sc



def up_sign_from_cameras(capture_dir, xyz):
    """+1 if 'up' is +z in PLY space, -1 if up is -z, None if unknown.
    Cameras are held above the floor, so up = from the bulk of the splats toward the cameras."""
    try:
        from colmap_io import read_images_binary, qvec_to_rotmat
        sparse = os.path.join(capture_dir, "sparse", "0", "images.bin")
        norm = os.path.join(capture_dir, "output", "normalization.json")
        T = np.array(json.load(open(norm))["transform"])
        centers = []
        for im in read_images_binary(sparse).values():
            w2c = np.eye(4); w2c[:3, :3] = qvec_to_rotmat(im.qvec); w2c[:3, 3] = im.tvec
            centers.append((T @ np.linalg.inv(w2c))[:3, 3])
        d = np.mean(np.array(centers)[:, 2]) - np.median(xyz[:, 2])
        return 1 if d > 0 else -1
    except Exception as e:
        print("  (could not infer up direction from cameras:", e, ")")
        return None


def find_floor(xyz, voxel, up_sign):
    """Return (point_on_plane, unit_normal_pointing_up) or None.
    Horizontal surfaces show as sharp peaks in the height histogram; the floor is the
    lowest significant one (a tabletop is higher). The plane is then refit with SVD so a
    slightly tilted floor is handled."""
    z = xyz[:, 2]
    bins = np.arange(z.min(), z.max() + voxel, voxel)
    cnt, edges = np.histogram(z, bins=bins)
    sm = ndimage.gaussian_filter1d(cnt.astype(float), 1.0)
    peaks = [i for i in range(1, len(sm) - 1)
             if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1] and sm[i] >= 0.15 * sm.max()]
    if not peaks:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    if up_sign is None:
        lo, hi = peaks[0], peaks[-1]
        pick = lo if sm[lo] >= sm[hi] else hi
        up_sign = 1 if pick == lo else -1
        print(f"  up direction unknown -> guessed {'+z' if up_sign == 1 else '-z'} from histogram")
    pick = peaks[0] if up_sign == 1 else peaks[-1]
    h0 = centers[pick]
    n = np.array([0.0, 0.0, 1.0])
    p0 = np.array([0.0, 0.0, h0])
    for _ in range(3):  # robust plane refit
        d = np.abs((xyz - p0) @ n)
        pts = xyz[d < 1.5 * voxel]
        if len(pts) < 100:
            break
        p0 = pts.mean(0)
        u, sv, vt = np.linalg.svd(pts - p0, full_matrices=False)
        n = vt[2]
    if n[2] * up_sign < 0:
        n = -n
    if abs(n[2]) < 0.85:
        print("  floor refit gave a non-horizontal plane; ignoring it")
        return None
    return p0, n


def floor_slab(p0, n, xyz, voxel, thick_up, pad):
    """Closed thin box on the floor plane covering all kept points."""
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, a); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    rel = xyz - p0
    uu, vv = rel @ u, rel @ v
    u0, u1, v0, v1 = uu.min() - pad, uu.max() + pad, vv.min() - pad, vv.max() + pad
    top = thick_up                       # slab top sits where the cleared region ends
    bot = thick_up - 3 * voxel
    corners = []
    for h in (bot, top):
        for (a_, b_) in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
            corners.append(p0 + a_ * u + b_ * v + h * n)
    V = np.array(corners)
    F = np.array([[0, 2, 1], [0, 3, 2],            # bottom
                  [4, 5, 6], [4, 6, 7],            # top
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return V, F


def build(xyz, op, voxel, sigma, thresh_frac, min_blob_frac, close_iters, min_part_voxels):
    pad = 4
    lo = xyz.min(0) - pad * voxel
    hi = xyz.max(0) + pad * voxel
    dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    if dims.prod() > 4e8:
        sys.exit(f"Grid too large {dims}; raise --voxel")
    idx = np.floor((xyz - lo) / voxel).astype(int)
    grid = np.zeros(dims, dtype=np.float32)
    np.add.at(grid, (idx[:, 0], idx[:, 1], idx[:, 2]), op.astype(np.float32))
    grid = ndimage.gaussian_filter(grid, sigma)
    occupied = grid[grid > 0]
    if occupied.size == 0:
        sys.exit("Empty density grid")
    level = thresh_frac * np.percentile(occupied, 95)
    solid = grid > level
    if close_iters > 0:
        solid = ndimage.binary_closing(solid, iterations=close_iters)
    lab, n = ndimage.label(solid, structure=np.ones((3, 3, 3)))
    if n == 0:
        sys.exit("Nothing above threshold; lower --thresh")
    sizes = ndimage.sum(solid, lab, range(1, n + 1))
    keep_ids = 1 + np.where((sizes >= min_blob_frac * sizes.max()) & (sizes >= min_part_voxels))[0]
    solid = np.isin(lab, keep_ids)
    parts = []
    for k, cid in enumerate(keep_ids):
        ii = np.argwhere(lab == cid)
        mn, mx = ii.min(0) * voxel + lo, (ii.max(0) + 1) * voxel + lo
        parts.append({"id": f"part_{k}", "voxels": int(sizes[cid - 1]),
                      "bbox_min": mn.tolist(), "bbox_max": mx.tolist(),
                      "size": (mx - mn).tolist(), "center": ((mn + mx) / 2).tolist()})
    print(f"  grid {tuple(dims)}  blobs {n} -> kept {len(keep_ids)} separate parts  solid voxels {int(solid.sum())}")
    # smooth binary field for marching cubes
    field = ndimage.gaussian_filter(solid.astype(np.float32), 0.8)
    verts, faces, _, _ = measure.marching_cubes(field, 0.5)
    verts = verts * voxel + lo
    return verts, faces, parts


def simplify(verts, faces, target):
    if len(faces) <= target:
        return verts, faces
    try:
        import open3d as o3d
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
        m = m.simplify_quadric_decimation(target)
        m.remove_degenerate_triangles(); m.remove_duplicated_triangles()
        m.remove_duplicated_vertices(); m.remove_non_manifold_edges()
        return np.asarray(m.vertices), np.asarray(m.triangles)
    except Exception as e:
        print("  (simplify skipped:", e, ")")
        return verts, faces


def orient_outward(verts, faces):
    # signed volume > 0 means outward-facing normals (CCW); flip if negative
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
    if vol < 0:
        faces = faces[:, ::-1]
    return faces, vol


def write_obj(path, groups):
    """groups: list of (name, verts, faces). One 'o' group per entry."""
    with open(path, "w") as f:
        f.write("# aerosense voxel collider\n")
        off = 0
        for name, verts, faces in groups:
            f.write(f"o {name}\n")
            for x, y, z in verts:
                f.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
            for a, b, c in faces + 1 + off:
                f.write(f"f {a} {b} {c}\n")
            off += len(verts)


def report(verts, faces):
    import trimesh
    m = trimesh.Trimesh(verts, faces, process=False)
    print("Collider report")
    print(f"  triangles {len(faces)}  vertices {len(verts)}")
    print(f"  bounds min {np.round(m.bounds[0],3)} max {np.round(m.bounds[1],3)}")
    print(f"  watertight {m.is_watertight}  components {len(m.split(only_watertight=False))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir")
    ap.add_argument("--ply")
    ap.add_argument("--ply-suffix", default="_clean")
    ap.add_argument("--out")
    ap.add_argument("--min-opacity", type=float, default=0.4)
    ap.add_argument("--no-floor", action="store_true", help="keep floor splats in the collider (old behaviour)")
    ap.add_argument("--floor-clear", type=float, default=2.0, help="voxels above the floor plane to cut away")
    ap.add_argument("--up-sign", type=int, choices=[-1, 1], default=0, help="+1/-1 if up is +z/-z in the PLY (0=auto)")
    ap.add_argument("--min-part-voxels", type=int, default=150, help="drop parts smaller than this many voxels")
    ap.add_argument("--voxel", type=float, default=0.0, help="0 = auto (bbox diag / 150)")
    ap.add_argument("--sigma", type=float, default=1.0, help="density smoothing, voxels")
    ap.add_argument("--thresh", type=float, default=0.35, help="fraction of 95th pct density")
    ap.add_argument("--min-blob-frac", type=float, default=0.02, help="drop blobs smaller than this fraction of largest")
    ap.add_argument("--close", type=int, default=1)
    ap.add_argument("--simplify-to", type=int, default=50000)
    a = ap.parse_args()

    if a.ply:
        ply = a.ply
        out = a.out or os.path.join(os.path.dirname(ply), "room_mesh.obj")
    elif a.capture_dir:
        ply = find_ply(a.capture_dir, a.ply_suffix)
        out = a.out or os.path.join(a.capture_dir, "room_mesh.obj")
    else:
        sys.exit("give --capture-dir or --ply")
    print("PLY:", ply)
    xyz, op, _ = load_splats(ply, a.min_opacity)
    print(f"  splats after opacity filter: {len(xyz)}")
    diag = np.linalg.norm(xyz.max(0) - xyz.min(0))
    voxel = a.voxel or diag / 150.0
    print(f"  voxel size {voxel:.4f}")
    floor = None
    if not a.no_floor:
        sign = a.up_sign or (up_sign_from_cameras(a.capture_dir, xyz) if a.capture_dir else None)
        if sign:
            print(f"  up direction: {'+z' if sign == 1 else '-z'} in PLY space")
        floor = find_floor(xyz, voxel, sign)
        if floor is None:
            print("  no clear floor found -> collider built without floor removal")
        else:
            p0, n = floor
            h = (xyz - p0) @ n
            clear = a.floor_clear * voxel
            keep = h > clear
            print(f"  floor plane found (normal {np.round(n,3)}); removed {int((~keep).sum())} floor/under-floor splats")
            xyz, op = xyz[keep], op[keep]
    verts, faces, parts = build(xyz, op, voxel, a.sigma, a.thresh, a.min_blob_frac, a.close, a.min_part_voxels)
    verts, faces = simplify(verts, faces, a.simplify_to)
    faces, vol = orient_outward(verts, np.asarray(faces))
    groups = [("parts", verts, faces)]
    if floor is not None:
        fv, ff = floor_slab(floor[0], floor[1], xyz, voxel, a.floor_clear * voxel, pad=0.05 * diag)
        groups.insert(0, ("floor", fv, ff))
    write_obj(out, groups)
    with open(os.path.splitext(out)[0] + "_parts.json", "w") as f:
        json.dump({"floor": floor is not None, "parts": parts}, f, indent=2)
    report(verts, faces)
    for p in parts:
        print(f"  {p['id']}: size {np.round(p['size'],3)}  center {np.round(p['center'],3)}")
    print("Wrote", out)


if __name__ == "__main__":
    main()