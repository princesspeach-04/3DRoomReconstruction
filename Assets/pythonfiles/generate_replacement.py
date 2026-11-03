"""
generate_replacement.py   (full rewrite)

For every object flagged treatment=generative_replace in objects_3d.json
(or the ids you pass with --ids), this script:

  1. Cuts that instance out of its own best frame using its YOLO mask
     (background -> neutral gray, square crop, ~85% foreground) so TripoSR
     sees a clean single object. No rembg download needed.
  2. Runs TripoSR ONCE for all crops (model loads once). Falls back to one
     call per object if the batch call fails.
  3. Solves the pose of the generated mesh automatically (class-agnostic):
       - world "up" comes from your camera poses (cameras are upright),
       - for each of the 6 possible canonical up-axes x a yaw sweep about
         world-up, the mesh is projected into the best frame with the REAL
         camera and compared with the YOLO mask (silhouette IoU) and with
         the image colours,
       - best candidate wins, then a finer yaw refinement.
     This removes the need to know TripoSR's axis convention.
  4. Scales uniformly (median of per-axis ratios) to the object's measured
     size from objects_3d.json and centres it exactly where the original
     reconstructed mesh was (object-local coordinates, same as the old OBJ).
  5. RE-COLOURS the mesh from your real photos: every vertex is projected into
     up to 4 well-spread frames (occlusion + facing + mask tests) and takes the
     real pixel colour. TripoSR's own blurry colours are only the fallback for
     vertices no photo can see. So the shape comes from TripoSR, the look from
     your video.
     Writes output/meshes/<id>_replacement.obj  (v x y z r g b  vertex colours)
     plus output/generated/<id>_align.png (debug: mask | render | overlap).
  6. Updates objects_3d.json: mesh_file -> replacement, keeps
     original_mesh_file, stores replace_region (PLY space) for the
     splat-removal step (replace_splats.py).

AUTOMATIC: with no --ids it processes every flagged object. Junk is skipped by itself:
  - objects with too few points, or that cover a large part of the scene (a whole-room
    "chair"/"bag" cluster), are skipped (guards),
  - objects whose generated mesh does not match the real silhouette (IoU < --min-iou)
    are rejected and keep their original mesh.
The best photo of each object is chosen automatically (large, sharp, not cut off).
Nothing is hardcoded to a class. A failure on one object never stops the run.

Usage:
  python generate_replacement.py --base-dir C:\\...\\video_Capture --triposr-dir C:\\...\\TripoSR
  python generate_replacement.py --base-dir ... --triposr-dir ... --ids laptop_0
  python generate_replacement.py --base-dir ... --triposr-dir ... --classes laptop phone
Needs (already in your venv): numpy, scipy, opencv. colmap_io.py must sit next to this file.
"""
import os
import re
import sys
import json
import glob
import shutil
import argparse
import subprocess
import numpy as np
import cv2
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_io import read_images_binary, read_cameras_binary, qvec_to_rotmat

GRID = 96            # silhouette comparison grid (longest side, pixels)
N_SAMPLES = 40000    # surface samples used for projection


# ----------------------------------------------------------------------------
# small geometry helpers
# ----------------------------------------------------------------------------
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)


def rot_between(a, b):
    """Rotation matrix taking unit vector a to unit vector b."""
    a = np.asarray(a, np.float64); a = a / np.linalg.norm(a)
    b = np.asarray(b, np.float64); b = b / np.linalg.norm(b)
    c = float(a @ b)
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        p = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0.0, 1.0, 0.0])
        p /= np.linalg.norm(p)
        return 2.0 * np.outer(p, p) - np.eye(3)
    vx = skew(np.cross(a, b))
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def axis_rot(axis, theta):
    axis = np.asarray(axis, np.float64); axis = axis / np.linalg.norm(axis)
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# ----------------------------------------------------------------------------
# OBJ io (own reader: independent of trimesh version, keeps "v x y z r g b")
# ----------------------------------------------------------------------------
def read_obj(path):
    V, C, F = [], [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                V.append([float(p[1]), float(p[2]), float(p[3])])
                if len(p) >= 7:
                    C.append([float(p[4]), float(p[5]), float(p[6])])
            elif line.startswith("f "):
                n = len(V)
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split("/")[0])
                    idx.append(i - 1 if i > 0 else n + i)
                for k in range(1, len(idx) - 1):
                    F.append([idx[0], idx[k], idx[k + 1]])
    V = np.array(V, dtype=np.float64).reshape(-1, 3)
    F = np.array(F, dtype=np.int64).reshape(-1, 3)
    Cn = None
    if len(C) == len(V) and len(C) > 0:
        Cn = np.array(C, dtype=np.float64)
        if Cn.max() > 1.5:
            Cn = Cn / 255.0
    return V, F, Cn


def write_obj(path, V, F, C):
    lines = ["# aerosense replacement mesh (object-local, PLY-space axes)"]
    if C is None:
        lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in V]
    else:
        lines += [f"v {x:.5f} {y:.5f} {z:.5f} {r:.4f} {g:.4f} {b:.4f}"
                  for (x, y, z), (r, g, b) in zip(V, C)]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in F]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def keep_main_components(V, F, C, frac=0.05):
    """Drop tiny floating islands (marching-cubes debris) from the generated mesh."""
    if len(F) == 0:
        return V, F, C
    n = len(V)
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    comp = lab[F[:, 0]]
    comp_area = np.bincount(comp, weights=area, minlength=nc)
    keep_comp = comp_area >= frac * comp_area.max()
    Fk = F[keep_comp[comp]]
    used = np.unique(Fk)
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[Fk], (C[used] if C is not None else None)


def fix_winding(V, F):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", v0 - V.mean(0), np.cross(v1 - V.mean(0), v2 - V.mean(0))).sum() / 6.0
    return (F[:, ::-1].copy() if vol < 0 else F), vol


def sample_surface(V, F, C, n, rng):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tot = area.sum()
    if tot <= 0:
        return V.copy(), (C.copy() if C is not None else np.full((len(V), 3), 0.7))
    idx = rng.choice(len(F), size=n, p=area / tot)
    r1 = np.sqrt(rng.random(n)); r2 = rng.random(n)
    w0, w1, w2 = 1 - r1, r1 * (1 - r2), r1 * r2
    P = a[idx] * w0[:, None] + b[idx] * w1[:, None] + c[idx] * w2[:, None]
    if C is None:
        PC = np.full((n, 3), 0.7)
    else:
        PC = C[F[idx, 0]] * w0[:, None] + C[F[idx, 1]] * w1[:, None] + C[F[idx, 2]] * w2[:, None]
    return P, PC


# ----------------------------------------------------------------------------
# cameras
# ----------------------------------------------------------------------------
def load_poses(base_dir):
    sparse = os.path.join(base_dir, "sparse", "0")
    images = read_images_binary(os.path.join(sparse, "images.bin"))
    cameras = read_cameras_binary(os.path.join(sparse, "cameras.bin"))
    with open(os.path.join(base_dir, "output", "normalization.json")) as f:
        T = np.array(json.load(f)["transform"])
    poses = {}
    for name, im in images.items():
        w2c = np.eye(4)
        w2c[:3, :3] = qvec_to_rotmat(im.qvec)
        w2c[:3, 3] = im.tvec
        c2w = T @ np.linalg.inv(w2c)
        c2w[:3, :3] /= np.linalg.norm(c2w[:3, 0])
        poses[name] = (c2w, cameras[im.camera_id])
    return poses


def world_up_from_cameras(poses):
    """Camera +y points DOWN (COLMAP/OpenCV), handheld video is upright."""
    ups = np.array([-c2w[:3, 1] for c2w, _ in poses.values()])
    up = ups.mean(0)
    return up / np.linalg.norm(up)


# ----------------------------------------------------------------------------
# step 1: crop for TripoSR
# ----------------------------------------------------------------------------
def make_triposr_crop(img_bgr, mask, fg_ratio, erode_px, out_size=512):
    m = mask.astype(np.uint8)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        e = cv2.erode(m, k)
        if e.sum() > 0.5 * m.sum():
            m = e
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pw, ph = x1 - x0 + 1, y1 - y0 + 1
    patch = img_bgr[y0:y1 + 1, x0:x1 + 1].copy()
    pm = m[y0:y1 + 1, x0:x1 + 1].astype(bool)
    patch[~pm] = 127
    side = int(np.ceil(max(pw, ph) / fg_ratio))
    canvas = np.full((side, side, 3), 127, dtype=np.uint8)
    oy, ox = (side - ph) // 2, (side - pw) // 2
    canvas[oy:oy + ph, ox:ox + pw] = patch
    canvas = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)
    H, W = m.shape
    touches = bool(x0 <= 1 or y0 <= 1 or x1 >= W - 2 or y1 >= H - 2)
    return canvas, touches, (int(pw), int(ph))


# ----------------------------------------------------------------------------
# step 2: TripoSR
# ----------------------------------------------------------------------------
def triposr_help(run_py, cwd):
    try:
        r = subprocess.run([sys.executable, run_py, "-h"], capture_output=True, text=True, cwd=cwd)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        print(f"  (could not read TripoSR help: {e})")
        return ""


def find_mesh(out_dir):
    for pat in ("mesh.obj", "*.obj"):
        c = glob.glob(os.path.join(out_dir, "**", pat), recursive=True)
        if c:
            return c[0]
    return None


def run_triposr(crop_paths, triposr_dir, batch_dir, mc_resolution):
    """Returns list (same order as crop_paths) of mesh paths or None."""
    run_py = os.path.join(triposr_dir, "run.py")
    if not os.path.exists(run_py):
        print(f"ERROR: {run_py} not found")
        return [None] * len(crop_paths)
    help_txt = triposr_help(run_py, triposr_dir)
    flags = []
    if "--no-remove-bg" in help_txt:
        flags.append("--no-remove-bg")
    else:
        print("  WARNING: this TripoSR run.py has no --no-remove-bg; it may run its own background removal")
    if "--model-save-format" in help_txt:
        flags += ["--model-save-format", "obj"]
    if "--mc-resolution" in help_txt:
        flags += ["--mc-resolution", str(mc_resolution)]

    if os.path.exists(batch_dir):
        shutil.rmtree(batch_dir)
    os.makedirs(batch_dir)

    print(f"\nRunning TripoSR on {len(crop_paths)} crop(s) in one call ...")
    r = subprocess.run([sys.executable, run_py] + list(crop_paths) + ["--output-dir", batch_dir] + flags,
                       cwd=triposr_dir)
    results = []
    if r.returncode == 0:
        for i in range(len(crop_paths)):
            results.append(find_mesh(os.path.join(batch_dir, str(i))))
        if all(results):
            return results
        print("  batch call finished but some meshes are missing -> retrying those one by one")
    else:
        print(f"  batch call failed (exit {r.returncode}) -> retrying one by one")
        results = [None] * len(crop_paths)

    for i, cp in enumerate(crop_paths):
        if i < len(results) and results[i]:
            continue
        od = os.path.join(batch_dir, f"single_{i}")
        os.makedirs(od, exist_ok=True)
        rr = subprocess.run([sys.executable, run_py, cp, "--output-dir", od] + flags, cwd=triposr_dir)
        results[i] = find_mesh(od) if rr.returncode == 0 else None
        if results[i] is None:
            print(f"  TripoSR failed for crop {os.path.basename(cp)}")
    return results


# ----------------------------------------------------------------------------
# step 3: pose solve (silhouette + colour against the real camera)
# ----------------------------------------------------------------------------
class Reference:
    def __init__(self, img_bgr, mask):
        ys, xs = np.nonzero(mask)
        self.x0, self.x1, self.y0, self.y1 = xs.min(), xs.max(), ys.min(), ys.max()
        w, h = self.x1 - self.x0 + 1, self.y1 - self.y0 + 1
        sc = GRID / max(w, h)
        self.gw, self.gh = max(8, int(round(w * sc))), max(8, int(round(h * sc)))
        mm = cv2.resize(mask[self.y0:self.y1 + 1, self.x0:self.x1 + 1].astype(np.float32),
                        (self.gw, self.gh), interpolation=cv2.INTER_AREA)
        self.mask = mm > 0.5
        rgb = cv2.cvtColor(img_bgr[self.y0:self.y1 + 1, self.x0:self.x1 + 1], cv2.COLOR_BGR2RGB)
        self.rgb = cv2.resize(rgb, (self.gw, self.gh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


class PoseSolver:
    def __init__(self, P, PC, s0, center_world, c2w, cam, img_wh, ref):
        self.P, self.PC, self.s0, self.center, self.ref = P, PC, s0, center_world, ref
        w2c = np.linalg.inv(c2w)
        self.Rw, self.tw = w2c[:3, :3], w2c[:3, 3]
        W, H = img_wh
        self.fx, self.fy = cam.fx * W / cam.width, cam.fy * H / cam.height
        self.cx, self.cy = cam.cx * W / cam.width, cam.cy * H / cam.height

    def render(self, R):
        pw = (self.P @ R.T) * self.s0 + self.center
        pc = pw @ self.Rw.T + self.tw
        z = pc[:, 2]
        ok = z > 1e-6
        if ok.sum() < 50:
            return None
        pc, cols, z = pc[ok], self.PC[ok], z[ok]
        u = self.fx * pc[:, 0] / z + self.cx
        v = self.fy * pc[:, 1] / z + self.cy
        gw, gh = self.ref.gw, self.ref.gh
        pw_, ph_ = max(u.max() - u.min(), 1e-6), max(v.max() - v.min(), 1e-6)
        k = min(gw / pw_, gh / ph_)
        gx = np.floor((u - (u.max() + u.min()) / 2) * k + gw / 2).astype(np.int64)
        gy = np.floor((v - (v.max() + v.min()) / 2) * k + gh / 2).astype(np.int64)
        inb = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
        gx, gy, cols, z = gx[inb], gy[inb], cols[inb], z[inb]
        order = np.argsort(-z)                 # far first, near overwrites
        gx, gy, cols = gx[order], gy[order], cols[order]
        rgb = np.zeros((gh, gw, 3), np.float32)
        hit = np.zeros((gh, gw), bool)
        rgb[gy, gx] = cols
        hit[gy, gx] = True
        sil = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        sil = ndimage.binary_fill_holes(sil)
        return sil, hit, rgb

    def score(self, R):
        out = self.render(R)
        if out is None:
            return -9.0, 0.0, 1.0, None
        sil, hit, rgb = out
        ref = self.ref
        union = (sil | ref.mask).sum()
        iou = float((sil & ref.mask).sum() / max(union, 1))
        region = hit & ref.mask
        cdiff = float(np.abs(rgb[region] - ref.rgb[region]).mean()) if region.sum() >= 20 else 0.5
        return iou - 0.7 * cdiff, iou, cdiff, out


def solve_pose(solver, up, yaw_step_deg):
    canon = [np.array(a, float) for a in
             ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]
    Ra_list = [rot_between(a, up) for a in canon]
    results = []
    for ai, Ra in enumerate(Ra_list):
        for th in np.arange(0.0, 360.0, yaw_step_deg):
            R = axis_rot(up, np.deg2rad(th)) @ Ra
            sc, iou, cd, _ = solver.score(R)
            results.append((sc, ai, th))
    results.sort(key=lambda t: -t[0])
    best = None
    for sc, ai, th in results[:6]:
        for d in np.arange(-yaw_step_deg, yaw_step_deg + 1e-6, yaw_step_deg / 5.0):
            t2 = th + d
            R = axis_rot(up, np.deg2rad(t2)) @ Ra_list[ai]
            s2, iou, cd, out = solver.score(R)
            if best is None or s2 > best["score"]:
                best = {"score": s2, "iou": iou, "cdiff": cd, "R": R, "yaw_deg": float(t2 % 360.0),
                        "axis_index": ai, "render": out}
    return best


def save_debug(path, ref, best):
    sil, hit, rgb = best["render"]
    a = ref.rgb * ref.mask[..., None]
    over = np.zeros((ref.gh, ref.gw, 3), np.float32)
    over[..., 0] = sil
    over[..., 1] = ref.mask
    tile = np.concatenate([a, rgb, over], axis=1)
    tile = (np.clip(tile, 0, 1) * 255).astype(np.uint8)
    tile = cv2.resize(tile, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))


# ----------------------------------------------------------------------------
# guards, automatic view selection, photo colouring
# ----------------------------------------------------------------------------
def scene_stats(base):
    try:
        from plyfile import PlyData
        cands = glob.glob(os.path.join(base, "output", "ply", "point_cloud_*_clean.ply"))
        if not cands:
            return None

        def step(p):
            m = re.search(r"point_cloud_(\d+)", os.path.basename(p))
            return int(m.group(1)) if m else -1
        p = max(cands, key=step)
        v = PlyData.read(p)["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], 1)
        lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
        return {"n": len(xyz), "diag": float(np.linalg.norm(hi - lo)), "ply": p}
    except Exception as e:
        print(f"  (scene guard disabled: {e})")
        return None


def guard_reason(obj, stats, a):
    pc = obj.get("point_count", 0)
    if pc < a.min_points:
        return f"only {pc} points (< {a.min_points}); too fragmentary to replace"
    if stats:
        if pc > a.max_scene_frac * stats["n"]:
            return (f"{pc} points = {100.0 * pc / stats['n']:.0f}% of the whole splat; this is a scene-sized "
                    f"cluster, not an object")
        if max(obj["size"]) > a.max_size_frac * stats["diag"]:
            return f"size {max(obj['size']):.2f} is more than {a.max_size_frac:.0%} of the scene extent"
    return None


def object_center(obj):
    pos = np.array(obj["position"], float)
    size = np.maximum(np.array(obj["size"], float), 1e-4)
    orig_path = obj.get("original_mesh_file") or obj.get("mesh_file")
    c_loc = np.zeros(3)
    if orig_path and os.path.exists(orig_path):
        ov, _, _ = read_obj(orig_path)
        if len(ov):
            c_loc = (ov.max(0) + ov.min(0)) / 2
    return pos, size, c_loc, pos + c_loc


def select_views(obj, center_world, detections, poses, images_dir, masks_dir, max_views, min_sep_deg=12.0):
    """Frames where THIS object (its centre projects inside a mask of its class) is large,
    sharp and not cut off by the image border. Returns [{frame, key, score}], best first."""
    cls = obj["class"]
    cands = []
    for fname, ents in detections.items():
        if fname not in poses:
            continue
        ents_c = [e for e in ents if e["class"] == cls]
        if not ents_c:
            continue
        npz_path = os.path.join(masks_dir, os.path.splitext(fname)[0] + ".npz")
        if not os.path.exists(npz_path):
            continue
        c2w, cam = poses[fname]
        w2c = np.linalg.inv(c2w)
        pc = w2c[:3, :3] @ center_world + w2c[:3, 3]
        if pc[2] <= 1e-6:
            continue
        masks = np.load(npz_path)
        for e in ents_c:
            m = masks[e["mask_key"]]
            Hm, Wm = m.shape
            ui = int(round((cam.fx * pc[0] / pc[2] + cam.cx) * Wm / cam.width))
            vi = int(round((cam.fy * pc[1] / pc[2] + cam.cy) * Hm / cam.height))
            if not (0 <= ui < Wm and 0 <= vi < Hm):
                continue
            r = 6
            if not m[max(vi - r, 0):vi + r + 1, max(ui - r, 0):ui + r + 1].any():
                continue
            ys, xs = np.nonzero(m)
            touches = bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= Wm - 2 or ys.max() >= Hm - 2)
            cands.append({"frame": fname, "key": e["mask_key"], "area": int(len(xs)), "touches": touches,
                          "cam_pos": c2w[:3, 3].copy()})
    cands.sort(key=lambda c: -c["area"])
    cands = cands[:15]
    for c in cands:
        gray = cv2.imread(os.path.join(images_dir, c["frame"]), cv2.IMREAD_GRAYSCALE)
        m = np.load(os.path.join(masks_dir, os.path.splitext(c["frame"])[0] + ".npz"))[c["key"]]
        if gray is None:
            c["score"] = 0.0
            continue
        if m.shape != gray.shape:
            m = cv2.resize(m.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        sharp = float(lap[m].var()) if m.any() else 0.0
        c["score"] = float(np.sqrt(c["area"]) * np.log1p(sharp) * (0.3 if c["touches"] else 1.0))
    cands.sort(key=lambda c: -c["score"])
    chosen = []
    for c in cands:
        d = c["cam_pos"] - center_world
        d = d / max(np.linalg.norm(d), 1e-9)
        ok = True
        for o in chosen:
            d2 = o["cam_pos"] - center_world
            d2 = d2 / max(np.linalg.norm(d2), 1e-9)
            if np.degrees(np.arccos(np.clip(d @ d2, -1, 1))) < min_sep_deg:
                ok = False
                break
        if ok:
            chosen.append(c)
        if len(chosen) >= max_views:
            break
    return [{"frame": c["frame"], "key": c["key"], "score": round(c["score"], 2), "touches": c["touches"],
             "cam_pos": c["cam_pos"]} for c in chosen]


def vertex_normals(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def photo_colors(Vw, F, base_colors, view_data, size_ref):
    """Colour every vertex from the real photos. view_data: [{rgb(float HxWx3), mask, c2w, cam}].
    Returns (colors Nx3 in 0-1, fraction of vertices coloured from photos)."""
    N = len(Vw)
    normals = vertex_normals(Vw, F)
    acc = np.zeros((N, 3))
    wsum = np.zeros(N)
    for vd in view_data:
        H, W = vd["rgb"].shape[:2]
        cam = vd["cam"]
        w2c = np.linalg.inv(vd["c2w"])
        pc = Vw @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        fx, fy = cam.fx * W / cam.width, cam.fy * H / cam.height
        cx, cy = cam.cx * W / cam.width, cam.cy * H / cam.height
        zs = np.maximum(z, 1e-6)
        u = fx * pc[:, 0] / zs + cx
        v = fy * pc[:, 1] / zs + cy
        ok = (z > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        if ok.sum() < 20:
            continue
        # occlusion: coarse z-buffer made from the mesh's own vertices
        ds = 0.25
        gw, gh = max(int(W * ds), 2), max(int(H * ds), 2)
        gx = np.clip((u * ds).astype(np.int64), 0, gw - 1)
        gy = np.clip((v * ds).astype(np.int64), 0, gh - 1)
        zbuf = np.full((gh, gw), np.inf)
        np.minimum.at(zbuf, (gy[ok], gx[ok]), z[ok])
        zbuf = ndimage.minimum_filter(zbuf, size=3)
        tol = 0.01 * size_ref + 10.0 * float(np.median(z[ok])) / fx
        visible = ok & (z <= zbuf[gy, gx] + tol)
        # facing the camera
        cam_pos = vd["c2w"][:3, 3]
        vdir = cam_pos - Vw
        vdir /= np.maximum(np.linalg.norm(vdir, axis=1, keepdims=True), 1e-12)
        cosang = (normals * vdir).sum(1)
        facing = cosang > 0.25
        # inside this object's own mask (eroded: no background / edge bleed)
        m = cv2.erode(vd["mask"].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        ui = np.clip(np.round(u).astype(np.int64), 0, W - 1)
        vi = np.clip(np.round(v).astype(np.int64), 0, H - 1)
        in_mask = m[vi, ui]
        use = visible & facing & in_mask
        if use.sum() == 0:
            continue
        col = cv2.remap(vd["rgb"], u.astype(np.float32).reshape(-1, 1), v.astype(np.float32).reshape(-1, 1),
                        cv2.INTER_LINEAR).reshape(-1, 3)
        w = np.where(use, np.clip(cosang, 0, 1) ** 4, 0.0)
        acc += col * w[:, None]
        wsum += w
    covered = wsum > 1e-6
    out = base_colors.copy() if base_colors is not None else np.full((N, 3), 0.6)
    out[covered] = acc[covered] / wsum[covered, None]
    return out, float(covered.mean())


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--triposr-dir", required=True)
    ap.add_argument("--ids", nargs="+", default=None,
                    help="Force these object ids (ignores the treatment label).")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="Restrict to these classes (default: every flagged object).")
    ap.add_argument("--force", action="store_true",
                    help="Redo objects that already have a replacement.")
    ap.add_argument("--crop-fg-ratio", type=float, default=0.85)
    ap.add_argument("--mask-erode", type=int, default=2, help="px to shrink the mask (removes background halo)")
    ap.add_argument("--min-mask-px", type=int, default=40, help="skip objects whose mask is smaller than this")
    ap.add_argument("--mc-resolution", type=int, default=256)
    ap.add_argument("--yaw-step", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-iou", type=float, default=0.5,
                    help="reject a replacement whose silhouette matches the real mask worse than this")
    ap.add_argument("--accept-low", action="store_true", help="keep replacements even below --min-iou")
    ap.add_argument("--no-photo-colors", action="store_true",
                    help="keep TripoSR's own vertex colours instead of colouring from your photos")
    ap.add_argument("--max-color-views", type=int, default=4)
    ap.add_argument("--min-points", type=int, default=800, help="guard: skip objects with fewer points")
    ap.add_argument("--max-scene-frac", type=float, default=0.2,
                    help="guard: skip objects holding more than this fraction of all splats")
    ap.add_argument("--max-size-frac", type=float, default=0.45,
                    help="guard: skip objects larger than this fraction of the scene extent")
    ap.add_argument("--no-guard", action="store_true", help="disable the three guards above")
    args = ap.parse_args()

    args.triposr_dir = os.path.abspath(args.triposr_dir)
    base = os.path.abspath(args.base_dir)
    seg_dir = os.path.join(base, "segmentation")
    images_dir = os.path.join(base, "images")
    masks_dir = os.path.join(seg_dir, "masks")
    mesh_dir = os.path.join(base, "output", "meshes")
    gen_dir = os.path.join(base, "output", "generated")
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    manifest_path = os.path.join(seg_dir, "objects_3d.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    if args.ids:
        todo = [o for o in manifest if o["id"] in set(args.ids)]
        missing = set(args.ids) - {o["id"] for o in todo}
        if missing:
            print(f"WARNING: ids not in manifest: {sorted(missing)}")
    else:
        todo = [o for o in manifest if o.get("treatment") == "generative_replace"]
    if args.classes:
        todo = [o for o in todo if o["class"] in args.classes]
    if not args.force:
        skipped = [o["id"] for o in todo if o.get("mesh_source") == "generative_replace_triposr"]
        if skipped:
            print(f"Already replaced (use --force to redo): {skipped}")
        todo = [o for o in todo if o.get("mesh_source") != "generative_replace_triposr"]
    print(f"{len(todo)} object(s) queued for generative replacement")
    if not todo:
        return

    poses = load_poses(base)
    up = world_up_from_cameras(poses)
    print(f"World up (PLY space) from cameras: {np.round(up, 3)}")
    with open(os.path.join(seg_dir, "detections_seg.json")) as f:
        detections = json.load(f)

    if not args.no_guard:
        stats = scene_stats(base)
        if stats:
            print(f"Scene: {stats['n']} splats, extent ~{stats['diag']:.2f}")
        kept = []
        for o in todo:
            why = guard_reason(o, stats, args)
            if why:
                print(f"SKIP {o['id']} ({o['class']}): {why}")
            else:
                kept.append(o)
        todo = kept
        if not todo:
            print("\nNothing left after the guards (use --no-guard to override).")
            return

    # ---- stage A: pick the best photos, make crops ----
    jobs = []
    for obj in todo:
        oid = obj["id"]
        pos, size, c_loc, center_world = object_center(obj)
        views = select_views(obj, center_world, detections, poses, images_dir, masks_dir, args.max_color_views)
        if not views and obj.get("best_frame") and obj.get("best_mask_key"):
            views = [{"frame": obj["best_frame"], "key": obj["best_mask_key"], "score": 0.0, "touches": False,
                      "cam_pos": poses[obj["best_frame"]][0][:3, 3]}] if obj["best_frame"] in poses else []
        print(f"\n{oid} (class={obj['class']}): {len(views)} usable view(s): "
              f"{[v['frame'] for v in views]}")
        if not views:
            print("  no usable frame/mask for this object, skipping")
            continue
        main = views[0]
        img_path = os.path.join(images_dir, main["frame"])
        npz_path = os.path.join(masks_dir, os.path.splitext(main["frame"])[0] + ".npz")
        if not (os.path.exists(img_path) and os.path.exists(npz_path)):
            print("  missing frame or mask file, skipping")
            continue
        img = cv2.imread(img_path)
        mask = np.load(npz_path)[main["key"]]
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
        mask = mask.astype(bool)
        if mask.sum() == 0:
            print("  empty mask, skipping")
            continue
        ys, xs = np.nonzero(mask)
        if min(xs.max() - xs.min(), ys.max() - ys.min()) < args.min_mask_px:
            print(f"  mask too small (<{args.min_mask_px}px), TripoSR would hallucinate, skipping")
            continue
        crop, touches, _ = make_triposr_crop(img, mask, args.crop_fg_ratio, args.mask_erode)
        if touches:
            print("  WARNING: mask touches the image border (object cut off) -> result may be poor")
        crop_path = os.path.join(gen_dir, f"{oid}_crop.png")
        cv2.imwrite(crop_path, crop)
        jobs.append({"obj": obj, "img": img, "mask": mask, "crop_path": crop_path, "touches": touches,
                     "views": views})
        print(f"  crop from {main['frame']} -> {crop_path}")

    if not jobs:
        print("\nNothing to generate.")
        return

    # ---- stage B: TripoSR ----
    mesh_paths = run_triposr([j["crop_path"] for j in jobs], args.triposr_dir,
                             os.path.join(gen_dir, "_triposr_batch"), args.mc_resolution)

    # ---- stage C: align, scale, write ----
    rng = np.random.default_rng(args.seed)
    ok_count, fail_count = 0, 0
    for job, mpath in zip(jobs, mesh_paths):
        obj = job["obj"]
        oid = obj["id"]
        print(f"\n{oid}: post-processing")
        if not mpath:
            print("  no mesh generated, keeping original mesh")
            fail_count += 1
            continue
        try:
            V, F, C = read_obj(mpath)
            if len(V) < 50 or len(F) < 50:
                raise RuntimeError("generated mesh is empty")
            V, F, C = keep_main_components(V, F, C)
            F, _ = fix_winding(V, F)
            print(f"  generated mesh: {len(V)} verts, {len(F)} tris, "
                  f"vertex colours: {'yes' if C is not None else 'NO (will be gray)'}")

            # unit-size, bbox-centred
            ctr = (V.max(0) + V.min(0)) / 2
            Vu = (V - ctr) / max((V.max(0) - V.min(0)).max(), 1e-9)

            # original object-local bbox centre (where the old mesh sat relative to 'position')
            pos, size, c_loc, center_world = object_center(obj)

            bf = job["views"][0]["frame"]
            c2w, cam = poses[bf]
            H, W = job["img"].shape[:2]
            ref = Reference(job["img"], job["mask"])
            P, PC = sample_surface(Vu, F, C, N_SAMPLES, rng)
            solver = PoseSolver(P, PC, float(size.max()), center_world, c2w, cam, (W, H), ref)
            best = solve_pose(solver, up, args.yaw_step)
            if best is None or best["render"] is None:
                raise RuntimeError("pose solve failed (object behind camera?)")
            print(f"  pose: axis#{best['axis_index']} yaw={best['yaw_deg']:.0f} deg  "
                  f"silhouette IoU={best['iou']:.2f}  colour diff={best['cdiff']:.2f}")
            low_conf = best["iou"] < 0.6
            save_debug(os.path.join(gen_dir, f"{oid}_align.png"), ref, best)
            if best["iou"] < args.min_iou and not args.accept_low:
                print(f"  REJECTED: generated shape does not match the real silhouette "
                      f"(IoU {best['iou']:.2f} < {args.min_iou}); keeping the original mesh. "
                      f"(see {oid}_align.png; --accept-low to keep anyway)")
                fail_count += 1
                continue
            if low_conf:
                print("  WARNING: weak silhouette match -> check the *_align.png debug image")

            vr = Vu @ best["R"].T
            ext = np.maximum(vr.max(0) - vr.min(0), 1e-6)
            ratios = size / ext
            s = float(np.median(ratios))
            spread = float(ratios.max() / max(ratios.min(), 1e-9))
            vs = vr * s
            vs = vs - (vs.max(0) + vs.min(0)) / 2 + c_loc
            print(f"  scale {s:.4f} (axis-ratio spread x{spread:.2f}), size {np.round(vs.max(0) - vs.min(0), 3)} "
                  f"vs measured {np.round(size, 3)}")

            photo_cov = None
            if not args.no_photo_colors:
                view_data = []
                for vw in job["views"]:
                    im = cv2.imread(os.path.join(images_dir, vw["frame"]))
                    mk = np.load(os.path.join(masks_dir, os.path.splitext(vw["frame"])[0] + ".npz"))[vw["key"]]
                    if im is None:
                        continue
                    if mk.shape != im.shape[:2]:
                        mk = cv2.resize(mk.astype(np.uint8), (im.shape[1], im.shape[0]),
                                        interpolation=cv2.INTER_NEAREST) > 0
                    c2w_v, cam_v = poses[vw["frame"]]
                    view_data.append({"rgb": cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
                                      "mask": mk.astype(bool), "c2w": c2w_v, "cam": cam_v})
                C, photo_cov = photo_colors(pos + vs, F, C, view_data, float(size.max()))
                print(f"  photo colours: {100 * photo_cov:.0f}% of vertices coloured from {len(view_data)} real view(s) "
                      f"(rest keep TripoSR's colour)")

            out_obj = os.path.join(mesh_dir, f"{oid}_replacement.obj")
            write_obj(out_obj, vs, F, C)

            if not obj.get("original_mesh_file"):
                obj["original_mesh_file"] = obj.get("mesh_file")
            obj["mesh_file"] = out_obj
            obj["mesh_source"] = "generative_replace_triposr"
            obj["replacement_has_vertex_colors"] = C is not None
            obj["replacement_align"] = {"iou": round(best["iou"], 3), "colour_diff": round(best["cdiff"], 3),
                                        "yaw_deg": round(best["yaw_deg"], 1), "scale": round(s, 5),
                                        "scale_spread": round(spread, 3), "low_confidence": bool(low_conf),
                                        "crop_touches_border": bool(job["touches"]),
                                        "photo_color_coverage": None if photo_cov is None else round(photo_cov, 3),
                                        "source_frame": bf,
                                        "color_frames": [v["frame"] for v in job["views"]]}
            obj["world_up"] = [round(float(x), 5) for x in up]
            obj["replace_region"] = {"min": (pos + vs.min(0)).tolist(), "max": (pos + vs.max(0)).tolist()}
            print(f"  wrote {out_obj}")
            ok_count += 1
        except Exception as e:
            print(f"  FAILED ({type(e).__name__}: {e}); keeping original mesh")
            fail_count += 1

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Replaced: {ok_count}   failed/skipped: {fail_count}")
    print(f"Manifest updated: {manifest_path}")
    print("Check the *_align.png images in output\\generated (left=mask, middle=render, right=overlap).")
    print("Next: python replace_splats.py --base-dir <capture_dir>   (removes the old splats)")


if __name__ == "__main__":
    main()