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

VERSION = "v8 (YOLO masks drive crops, containment test, point masks off by default)"
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
    # TripoSR only creates <out>/<i>/ itself when it runs its own background removal;
    # with --no-remove-bg the export fails (FileNotFoundError) unless the folders exist.
    for i in range(len(crop_paths)):
        os.makedirs(os.path.join(batch_dir, str(i)), exist_ok=True)

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
        os.makedirs(os.path.join(od, "0"), exist_ok=True)
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


class MultiSolver:
    """Scores a candidate rotation against ALL selected views (silhouette IoU + colour) and
    against the measured object size. One steep top-down view cannot tell 'lying flat' from
    'standing on edge'; several views from different angles can."""

    def __init__(self, Vu, P, PC, size, center_world, views, size_weight=0.5):
        self.size = np.maximum(np.asarray(size, float), 1e-4)
        self.Vsub = Vu[:: max(1, len(Vu) // 4000)]
        self.size_weight = size_weight
        s0 = float(self.size.max())
        self.solvers = []
        for vw in views:
            H, W = vw["img"].shape[:2]
            self.solvers.append(PoseSolver(P, PC, s0, center_world, vw["c2w"], vw["cam"], (W, H),
                                           Reference(vw["img"], vw["mask"])))

    def size_term(self, R):
        Vr = self.Vsub @ R.T
        ext = np.maximum(Vr.max(0) - Vr.min(0), 1e-6)
        return float(np.log(self.size / ext).std())

    def score(self, R):
        scs, ious, cds, outs = [], [], [], []
        for sv in self.solvers:
            sc, iou, cd, out = sv.score(R)
            outs.append(out)
            if out is None:
                continue
            scs.append(sc); ious.append(iou); cds.append(cd)
        if not scs:
            return -9.0, 0.0, 1.0, outs
        st = self.size_term(R)
        return float(np.mean(scs)) - self.size_weight * st, float(np.mean(ious)), float(np.mean(cds)), outs


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
            s2, iou, cd, outs = solver.score(R)
            if best is None or s2 > best["score"]:
                best = {"score": s2, "iou": iou, "cdiff": cd, "R": R, "yaw_deg": float(t2 % 360.0),
                        "axis_index": ai, "render": outs}
    return best


def refine_placement(Vr_centered, s_init, center_init, size, solvers):
    """Fit uniform scale + 3D offset so the projected mesh bounding boxes match the real YOLO mask
    boxes in every selected view (mesh shape/rotation stay fixed). Soft priors keep it near the
    cluster-based guess when only one view constrains it."""
    from scipy.optimize import least_squares
    sub = Vr_centered[:: max(1, len(Vr_centered) // 6000)]
    unit = 0.1 * float(np.max(size))
    DW = 30.0   # a 3% box error in the image costs as much as the priors' 1-sigma

    def residual(p):
        t = p[:3] * unit
        s = s_init * np.exp(p[3])
        pw = sub * s + center_init + t
        res = []
        for sv in solvers:
            pc = pw @ sv.Rw.T + sv.tw
            z = pc[:, 2]
            if (z <= 1e-6).any():
                res += [10.0] * 4
                continue
            u = sv.fx * pc[:, 0] / z + sv.cx
            v = sv.fy * pc[:, 1] / z + sv.cy
            r = sv.ref
            D = float(max(r.x1 - r.x0, r.y1 - r.y0) + 1)
            res += [DW * (u.min() - r.x0) / D, DW * (u.max() - r.x1) / D,
                    DW * (v.min() - r.y0) / D, DW * (v.max() - r.y1) / D]
        res += list(p[:3] / 2.5)       # prior: offset within ~25% of object size
        res += [p[3] / 0.7]            # prior: scale within ~2x of the cluster estimate
        return np.array(res)

    sol = least_squares(residual, np.zeros(4), method="trf", max_nfev=200)
    t = sol.x[:3] * unit
    s = float(s_init * np.exp(sol.x[3]))
    r = residual(sol.x)[: 4 * len(solvers)] / DW
    return s, t, float(np.abs(r).mean())


def save_debug(path, solvers, outs):
    rows = []
    for sv, out in zip(solvers, outs):
        if out is None:
            continue
        ref = sv.ref
        sil, hit, rgb = out
        a = ref.rgb * ref.mask[..., None]
        over = np.zeros((ref.gh, ref.gw, 3), np.float32)
        over[..., 0] = sil
        over[..., 1] = ref.mask
        tile = np.concatenate([a, rgb, over], axis=1)
        rows.append((np.clip(tile, 0, 1) * 255).astype(np.uint8))
    if not rows:
        return
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 2), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    tile = cv2.resize(np.concatenate(rows, axis=0), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))


# ----------------------------------------------------------------------------
# guards, automatic view selection, photo colouring
# ----------------------------------------------------------------------------
def load_clean_xyz(base):
    """Splat centres of the newest *_clean.ply (None if unavailable)."""
    try:
        from plyfile import PlyData
        cands = glob.glob(os.path.join(base, "output", "ply", "point_cloud_*_clean.ply"))
        if not cands:
            return None, None

        def step(p):
            m = re.search(r"point_cloud_(\d+)", os.path.basename(p))
            return int(m.group(1)) if m else -1
        p = max(cands, key=step)
        v = PlyData.read(p)["vertex"]
        return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), p
    except Exception as e:
        print(f"  (could not read the clean PLY: {e})")
        return None, None


def scene_stats(xyz):
    if xyz is None or len(xyz) == 0:
        return None
    lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    return {"n": len(xyz), "diag": float(np.linalg.norm(hi - lo))}


def estimate_world_up(poses, xyz, rng):
    """World-up in PLY space.
    Camera 'up' is unreliable when the video looks DOWN at a table (image-up is then
    horizontal), so: if the splat scene is flat (a floor/table scene), up = its thin
    principal axis (the floor normal) and only the SIGN comes from the cameras (they
    are held above the objects). If the scene is not flat, fall back to camera 'up'."""
    cam_up = world_up_from_cameras(poses)
    if xyz is None or len(xyz) < 1000:
        return cam_up, "cameras (no splat data)"
    sub = xyz if len(xyz) <= 200000 else xyz[rng.choice(len(xyz), 200000, replace=False)]
    c = np.median(sub, axis=0)
    w, v = np.linalg.eigh(np.cov((sub - c).T))        # ascending
    ratio = float(w[0] / max(w[1], 1e-12))
    if ratio < 0.5:
        n = v[:, 0]
        cam_centers = np.array([p[0][:3, 3] for p in poses.values()])
        if float((cam_centers.mean(0) - c) @ n) < 0:
            n = -n
        return n / np.linalg.norm(n), f"flat-scene axis (variance ratio {ratio:.2f}), sign from camera positions"
    return cam_up, f"cameras (scene not flat, variance ratio {ratio:.2f})"


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


def bilinear_sample(img, u, v):
    """img HxWx3 float, u/v pixel coords (arrays). Plain numpy: cv2.remap refuses > 32767 points."""
    H, W = img.shape[:2]
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


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
        zbuf = np.where(np.isinf(zbuf), ndimage.minimum_filter(zbuf, size=3), zbuf)  # fill gaps only
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
        col = bilinear_sample(vd["rgb"], u, v)
        w = np.where(use, np.clip(cosang, 0, 1) ** 4, 0.0)
        acc += col * w[:, None]
        wsum += w
    covered = wsum > 1e-6
    out = base_colors.copy() if base_colors is not None else np.full((N, 3), 0.6)
    out[covered] = acc[covered] / wsum[covered, None]
    return out, float(covered.mean())


# ----------------------------------------------------------------------------
# virtual views: find every frame where the object's own 3D points are fully visible
# ----------------------------------------------------------------------------
def load_clean_splats(base):
    """(xyz, opacity) of the newest *_clean.ply, or (None, None)."""
    try:
        from plyfile import PlyData
        cands = glob.glob(os.path.join(base, "output", "ply", "point_cloud_*_clean.ply"))
        if not cands:
            return None, None

        def step(p):
            m = re.search(r"point_cloud_(\d+)", os.path.basename(p))
            return int(m.group(1)) if m else -1
        v = PlyData.read(max(cands, key=step))["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
        if "opacity" in v.data.dtype.names:
            op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))
        else:
            op = np.ones(len(xyz))
        return xyz, op
    except Exception as e:
        print(f"  (could not read the clean PLY: {e})")
        return None, None


def footprint_mask(u, v, W, H, scale=0.5, r=None, upscale=True):
    """Silhouette of a set of surface points (splat centres) projected at (u, v).
    Returns (mask, closing radius used). Pass the same r to compare two point sets fairly."""
    w, h = max(int(W * scale), 8), max(int(H * scale), 8)
    gx = np.clip((u * scale).astype(np.int64), 0, w - 1)
    gy = np.clip((v * scale).astype(np.int64), 0, h - 1)
    m = np.zeros((h, w), np.uint8)
    m[gy, gx] = 1
    if r is None:
        area = float((gx.max() - gx.min() + 1) * (gy.max() - gy.min() + 1))
        spacing = np.sqrt(max(area, 1.0) / max(len(gx), 1))
        r = int(np.clip(round(spacing * 1.3), 1, 14))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    m = ndimage.binary_fill_holes(m > 0).astype(np.uint8)
    if upscale:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return m > 0, r


def compute_virtual_views(objs, poses, xyz, opacity, up, images_dir, args, keep_per_obj=20):
    """For every object that has a saved point set: scan ALL frames, keep the ones where the object
    is fully inside the image and mostly unoccluded (first-visible-surface test), build its silhouette
    from the visible points, and rank by size x sharpness x viewing elevation.
    Returns {object_id: [candidate dicts, best first]}."""
    pts = {}
    for o in objs:
        pf = o.get("points_file")
        if pf and os.path.exists(pf):
            P = np.load(pf).astype(np.float32)
            if len(P) > 25000:
                P = P[np.random.default_rng(0).choice(len(P), 25000, replace=False)]
            pts[o["id"]] = P
    if not pts or xyz is None:
        return {}
    centers = {o["id"]: object_center(o)[3] for o in objs}
    solid = opacity >= 0.3
    cand = {oid: [] for oid in pts}
    names = sorted(poses.keys())
    for fi, fname in enumerate(names):
        c2w, cam = poses[fname]
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].astype(np.float32)
        t = w2c[:3, 3].astype(np.float32)
        W, H = int(cam.width), int(cam.height)
        pc = xyz @ R.T + t
        z = pc[:, 2]
        zs = np.clip(z, 1e-6, None)
        u = cam.fx * pc[:, 0] / zs + cam.cx
        v = cam.fy * pc[:, 1] / zs + cam.cy
        ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        d = 8.0
        gw, gh = max(int(W / d), 2), max(int(H / d), 2)
        gx = np.clip((u / d).astype(np.int64), 0, gw - 1)
        gy = np.clip((v / d).astype(np.int64), 0, gh - 1)
        sel = ok & solid
        zb = np.full((gh, gw), np.inf, np.float32)
        np.minimum.at(zb, (gy[sel], gx[sel]), z[sel])
        zb = np.where(np.isinf(zb), ndimage.minimum_filter(zb, size=3), zb)
        for oid, P in pts.items():
            q = P @ R.T + t
            zq = q[:, 2]
            if (zq > 1e-6).mean() < 0.97:
                continue
            zqs = np.clip(zq, 1e-6, None)
            uq = cam.fx * q[:, 0] / zqs + cam.cx
            vq = cam.fy * q[:, 1] / zqs + cam.cy
            m = 4.0
            inside = (zq > 1e-6) & (uq >= m) & (uq < W - m) & (vq >= m) & (vq < H - m)
            if inside.mean() < 0.97:
                continue
            qx = np.clip((uq / d).astype(np.int64), 0, gw - 1)
            qy = np.clip((vq / d).astype(np.int64), 0, gh - 1)
            vis = inside & (zq <= zb[qy, qx] + 0.03 + 0.02 * zq)
            if vis.sum() < 60:
                continue
            # Occlusion by OTHER things = the visible silhouette is smaller than the full one.
            # (The object's own hidden back side projects inside its front silhouette, so it
            #  must not count against the view.)
            sc_ = 0.25
            m_full, r_ = footprint_mask(uq[inside], vq[inside], W, H, scale=sc_, upscale=False)
            m_vis, _ = footprint_mask(uq[vis], vq[vis], W, H, scale=sc_, r=r_, upscale=False)
            frac_vis = float(m_vis.sum() / max(m_full.sum(), 1))
            if frac_vis < 0.85:
                continue
            ys0, xs0 = vq[vis], uq[vis]
            side = min(xs0.max() - xs0.min(), ys0.max() - ys0.min())
            if side < args.min_mask_px:
                continue
            cam_pos = c2w[:3, 3]
            dvec = cam_pos - centers[oid]
            elev = float(np.degrees(np.arcsin(np.clip(dvec @ up / max(np.linalg.norm(dvec), 1e-9), -1, 1))))
            area_est = float((xs0.max() - xs0.min()) * (ys0.max() - ys0.min()))
            ew = 1.0 if 8.0 <= elev <= 65.0 else 0.7
            pre = np.sqrt(area_est) * frac_vis * ew
            cand[oid].append({"frame": fname, "uv": (xs0.astype(np.float32), ys0.astype(np.float32)),
                              "WH": (W, H), "pre": pre, "frac_vis": frac_vis, "elev": elev, "ew": ew,
                              "cam_pos": cam_pos.copy()})
            if len(cand[oid]) > keep_per_obj:
                cand[oid].sort(key=lambda c: -c["pre"])
                del cand[oid][keep_per_obj:]
        if (fi + 1) % 100 == 0 or fi == len(names) - 1:
            print(f"  scanned {fi + 1}/{len(names)} frames for object visibility")
    out = {}
    for oid, lst in cand.items():
        lst.sort(key=lambda c: -c["pre"])
        for c in lst:
            gray = cv2.imread(os.path.join(images_dir, c["frame"]), cv2.IMREAD_GRAYSCALE)
            c["mask"], _ = footprint_mask(c["uv"][0], c["uv"][1], *c["WH"])
            if gray is None:
                c["score"] = 0.0
                continue
            mk = c["mask"]
            if mk.shape != gray.shape:
                mk = cv2.resize(mk.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                c["mask"] = mk
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            sharp = float(lap[mk].var()) if mk.any() else 0.0
            c["score"] = float(np.sqrt(mk.sum()) * np.log1p(sharp) * c["frac_vis"] * c["ew"])
        lst.sort(key=lambda c: -c["score"])
        out[oid] = lst
    return out


def pick_spread(cands, center_world, k, min_sep_deg=12.0):
    chosen = []
    for c in cands:
        d = c["cam_pos"] - center_world
        d = d / max(np.linalg.norm(d), 1e-9)
        if all(np.degrees(np.arccos(np.clip(d @ ((o["cam_pos"] - center_world) /
               max(np.linalg.norm(o["cam_pos"] - center_world), 1e-9)), -1, 1))) >= min_sep_deg for o in chosen):
            chosen.append(c)
        if len(chosen) >= k:
            break
    return chosen


def mask_iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def load_view(frame, mask, poses, images_dir, detections, cls, masks_dir, center_world=None,
              allow_point_masks=False, min_contain=0.6):
    """Image + mask for one view.

    The point-based silhouette `mask` comes from the object's 3D cluster. Clusters are often
    INFLATED (they swallow floor / neighbours), so that silhouette is only used as a rough
    search region: a YOLO mask of the same class is adopted when most of it lies INSIDE the
    silhouette (containment, not IoU - an inflated silhouette has a low IoU with the true
    object but still contains it). If several YOLO instances qualify, the one whose centre is
    closest to the projected object centre wins.
    Without a YOLO mask the view is dropped (returns None) unless allow_point_masks is set."""
    im = cv2.imread(os.path.join(images_dir, frame))
    if im is None:
        return None
    if mask is not None and mask.shape != im.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    c2w, cam = poses[frame]
    used, chosen = "points", None
    npz = os.path.join(masks_dir, os.path.splitext(frame)[0] + ".npz")
    if mask is not None and os.path.exists(npz):
        data = np.load(npz)
        pu = pv = None
        if center_world is not None:
            w2c = np.linalg.inv(c2w)
            pc = w2c[:3, :3] @ center_world + w2c[:3, 3]
            if pc[2] > 1e-6:
                pu = (cam.fx * pc[0] / pc[2] + cam.cx) * im.shape[1] / cam.width
                pv = (cam.fy * pc[1] / pc[2] + cam.cy) * im.shape[0] / cam.height
        best, bd = None, 1e18
        for e in detections.get(frame, []):
            if e["class"] != cls:
                continue
            ym = data[e["mask_key"]].astype(bool)
            if ym.shape != mask.shape:
                ym = cv2.resize(ym.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            n = int(ym.sum())
            if n < 50:
                continue
            if float((ym & mask).sum()) / n < min_contain:
                continue
            ys, xs = np.nonzero(ym)
            d = 0.0 if pu is None else float(np.hypot(xs.mean() - pu, ys.mean() - pv))
            if d < bd:
                best, bd = ym, d
        if best is not None:
            chosen, used = best, "yolo"
    if chosen is None:
        if not allow_point_masks:
            return None
        chosen = mask
    return {"frame": frame, "img": im, "mask": chosen.astype(bool), "c2w": c2w, "cam": cam, "mask_src": used}


def fit_candidate(mpath, size, center_world, vds, up, args, rng):
    """Load one TripoSR mesh and solve its pose against all views."""
    V, F, C = read_obj(mpath)
    if len(V) < 50 or len(F) < 50:
        raise RuntimeError("generated mesh is empty")
    V, F, C = keep_main_components(V, F, C)
    F, _ = fix_winding(V, F)
    ctr = (V.max(0) + V.min(0)) / 2
    Vu = (V - ctr) / max((V.max(0) - V.min(0)).max(), 1e-9)
    P, PC = sample_surface(Vu, F, C, N_SAMPLES, rng)
    ms = MultiSolver(Vu, P, PC, size, center_world, vds, size_weight=args.size_weight)
    best = solve_pose(ms, up, args.yaw_step)
    if best is None or all(o is None for o in best["render"]):
        raise RuntimeError("pose solve failed (object behind the camera?)")
    return {"V": V, "F": F, "C": C, "Vu": Vu, "ms": ms, "best": best, "mpath": mpath}


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
    ap.add_argument("--min-mask-px", type=int, default=90,
                    help="an object must be at least this many pixels wide/tall in a view to be used "
                         "(TripoSR hallucinates from tiny crops)")
    ap.add_argument("--candidates", type=int, default=3,
                    help="number of different crops (views) tried per object; the best-matching mesh wins")
    ap.add_argument("--mc-resolution", type=int, default=256)
    ap.add_argument("--yaw-step", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-iou", type=float, default=0.7,
                    help="reject a replacement whose mean silhouette IoU (over all views) is below this")
    ap.add_argument("--min-view-iou", type=float, default=0.55,
                    help="reject if the shape matches ANY selected view worse than this (views must agree)")
    ap.add_argument("--max-fit-error", type=float, default=0.08,
                    help="reject if, after fitting, the mesh box differs from the real mask boxes by more than "
                         "this fraction of the object size (views disagree about size/position)")
    ap.add_argument("--accept-low", action="store_true",
                    help="keep replacements even when they fail --min-iou / --min-view-iou / --max-fit-error")
    ap.add_argument("--no-photo-colors", action="store_true",
                    help="keep TripoSR's own vertex colours instead of colouring from your photos")
    ap.add_argument("--max-color-views", type=int, default=5)
    ap.add_argument("--min-points", type=int, default=800, help="guard: skip objects with fewer points")
    ap.add_argument("--max-scene-frac", type=float, default=0.2,
                    help="guard: skip objects holding more than this fraction of all splats")
    ap.add_argument("--max-size-frac", type=float, default=0.45,
                    help="guard: skip objects larger than this fraction of the scene extent")
    ap.add_argument("--no-guard", action="store_true", help="disable the three guards above")
    ap.add_argument("--scale-mode", choices=["image", "cluster"], default="image",
                    help="image: fit scale+position to the real masks in all views (default); "
                         "cluster: just match the measured cluster bounding box")
    ap.add_argument("--allow-point-masks", action="store_true",
                    help="also use views whose mask comes only from the 3D points (unreliable when clusters are inflated)")
    ap.add_argument("--size-weight", type=float, default=0.1,
                    help="how strongly the pose solve prefers the 3D-cluster size (0 = ignore it; clusters can be inflated)")
    ap.add_argument("--scale-ratio", nargs=2, type=float, default=[0.25, 3.0], metavar=("LO", "HI"),
                    help="LO: below this fitted/cluster size ratio only a note is printed (cluster inflated); "
                         "HI: above this the replacement is rejected")
    ap.add_argument("--no-virtual-views", action="store_true",
                    help="only use YOLO frames as views (old behaviour)")
    ap.add_argument("--up", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                    help="override the automatic world-up direction (PLY space)")
    args = ap.parse_args()

    print(f"generate_replacement.py {VERSION}  ({os.path.abspath(__file__)})")
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
    rng = np.random.default_rng(args.seed)
    splat_xyz, splat_op = load_clean_splats(base)
    clean_xyz = None if splat_xyz is None else splat_xyz.astype(np.float64)
    if args.up:
        up = np.array(args.up, float)
        up /= np.linalg.norm(up)
        how = "--up override"
    else:
        up, how = estimate_world_up(poses, clean_xyz, rng)
    print(f"World up (PLY space): {np.round(up, 3)}  [{how}]")
    with open(os.path.join(seg_dir, "detections_seg.json")) as f:
        detections = json.load(f)

    if not args.no_guard:
        stats = scene_stats(clean_xyz)
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

    # ---- stage A: views. Preferred: every frame where the object's own 3D points are fully visible ----
    virtual = {}
    if not args.no_virtual_views:
        have_pts = [o for o in todo if o.get("points_file") and os.path.exists(o["points_file"])]
        if have_pts:
            print(f"\nSearching all {len(poses)} frames for clean views of {len(have_pts)} object(s) ...")
            virtual = compute_virtual_views(have_pts, poses, splat_xyz, splat_op, up, images_dir, args)
        missing_pts = [o["id"] for o in todo if o["id"] not in {x["id"] for x in have_pts}]
        if missing_pts:
            print(f"  (no points_file for {missing_pts}: rerun build_object_meshes.py; using YOLO frames for them)")

    jobs = []
    for obj in todo:
        oid = obj["id"]
        pos, size, c_loc, center_world = object_center(obj)
        vds = []
        if oid in virtual and virtual[oid]:
            chosen = pick_spread(virtual[oid], center_world, args.max_color_views * 4)
            for c in chosen:
                if len(vds) >= args.max_color_views:
                    break
                vd = load_view(c["frame"], c["mask"], poses, images_dir, detections, obj["class"], masks_dir,
                               center_world=center_world, allow_point_masks=args.allow_point_masks)
                if vd is not None and vd["mask"].any():
                    vd["elev"] = round(c["elev"], 1)
                    vds.append(vd)
            kind = "3D-visible+YOLO"
        if not vds:
            if oid in virtual and virtual[oid]:
                print(f"  {oid}: no 3D-visible frame has a YOLO mask inside the object's footprint; "
                      f"falling back to YOLO-detected frames")
            vds = []
            kind = "YOLO-detected"
        if not vds:
            views = select_views(obj, center_world, detections, poses, images_dir, masks_dir, args.max_color_views)
            for vw in views:
                npz = os.path.join(masks_dir, os.path.splitext(vw["frame"])[0] + ".npz")
                if not os.path.exists(npz) or vw["frame"] not in poses:
                    continue
                mk = np.load(npz)[vw["key"]].astype(bool)
                im_ = cv2.imread(os.path.join(images_dir, vw["frame"]))
                if im_ is None:
                    continue
                c2w_, cam_ = poses[vw["frame"]]
                vds.append({"frame": vw["frame"], "img": im_, "mask": mk, "c2w": c2w_, "cam": cam_,
                            "mask_src": "yolo"})
            kind = "YOLO-detected"
        print(f"\n{oid} (class={obj['class']}): {len(vds)} {kind} view(s): "
              f"{[(v['frame'], v['mask_src'], v.get('elev')) for v in vds]}")
        if not vds:
            print("  never seen large/complete enough in any frame, skipping")
            continue
        crops = []
        for j, vd in enumerate(vds[:args.candidates]):
            crop, touches, _ = make_triposr_crop(vd["img"], vd["mask"], args.crop_fg_ratio, args.mask_erode)
            cp = os.path.join(gen_dir, f"{oid}_crop{j}.png")
            cv2.imwrite(cp, crop)
            crops.append({"path": cp, "frame": vd["frame"], "touches": touches})
        jobs.append({"obj": obj, "view_data": vds, "crops": crops})
        print(f"  {len(crops)} candidate crop(s) -> {gen_dir}\\{oid}_crop*.png")

    if not jobs:
        print("\nNothing to generate.")
        return

    # ---- stage B: TripoSR for every crop of every object in ONE call ----
    all_crops = [c["path"] for j in jobs for c in j["crops"]]
    mesh_paths = run_triposr(all_crops, args.triposr_dir, os.path.join(gen_dir, "_triposr_batch"), args.mc_resolution)
    k = 0
    for j in jobs:
        for c in j["crops"]:
            c["mesh"] = mesh_paths[k]
            k += 1

    # ---- stage C: fit every candidate, keep the one that matches reality best ----
    ok_count, fail_count = 0, 0
    for job in jobs:
        obj = job["obj"]
        oid = obj["id"]
        vds = job["view_data"]
        print(f"\n{oid}: fitting {len(job['crops'])} candidate mesh(es) against {len(vds)} view(s)")
        pos, size, c_loc, center_world = object_center(obj)
        fits = []
        for ci, c in enumerate(job["crops"]):
            if not c.get("mesh"):
                print(f"  candidate {ci} ({c['frame']}): TripoSR produced no mesh")
                continue
            try:
                fc = fit_candidate(c["mesh"], size, center_world, vds, up, args, rng)
            except Exception as e:
                print(f"  candidate {ci} ({c['frame']}): failed ({type(e).__name__}: {e})")
                continue
            fc["crop_index"], fc["crop_frame"], fc["touches"] = ci, c["frame"], c["touches"]
            b = fc["best"]
            print(f"  candidate {ci} ({c['frame']}): mean IoU {b['iou']:.2f}  colour diff {b['cdiff']:.2f}  "
                  f"match score {b['score']:.3f}")
            fits.append(fc)
        if not fits:
            print("  no usable candidate, keeping original mesh")
            fail_count += 1
            continue
        try:
            fc = max(fits, key=lambda x: x["best"]["score"])
            best, ms = fc["best"], fc["ms"]
            V, F, C, Vu, R = fc["V"], fc["F"], fc["C"], fc["Vu"], best["R"]
            per_view_iou = [round(float(sv.score(R)[1]), 2) if o is not None else None
                            for sv, o in zip(ms.solvers, best["render"])]
            print(f"  -> using candidate {fc['crop_index']} ({fc['crop_frame']}): axis#{best['axis_index']} "
                  f"yaw={best['yaw_deg']:.0f} deg  mean silhouette IoU={best['iou']:.2f} over {len(vds)} view(s) "
                  f"{per_view_iou}  colour diff={best['cdiff']:.2f}")
            low_conf = best["iou"] < 0.6
            save_debug(os.path.join(gen_dir, f"{oid}_align.png"), ms.solvers, best["render"])
            if best["iou"] < args.min_iou and not args.accept_low:
                print(f"  REJECTED: generated shape does not match the real silhouettes "
                      f"(IoU {best['iou']:.2f} < {args.min_iou}); keeping the original mesh. "
                      f"(see {oid}_align.png; --accept-low to keep anyway)")
                fail_count += 1
                continue
            worst = min([x for x in per_view_iou if x is not None], default=0.0)
            if worst < args.min_view_iou and not args.accept_low:
                print(f"  REJECTED: the views disagree (worst per-view IoU {worst:.2f} < {args.min_view_iou}); "
                      f"keeping the original mesh. (see {oid}_align.png; --accept-low to keep anyway)")
                fail_count += 1
                continue
            if low_conf:
                print("  WARNING: weak silhouette match -> check the *_align.png debug image")

            Vr = Vu @ R.T
            Vr_c = Vr - (Vr.max(0) + Vr.min(0)) / 2
            ext = np.maximum(Vr_c.max(0) - Vr_c.min(0), 1e-6)
            ratios = size / ext
            s_cluster = float(np.median(ratios))
            spread = float(ratios.max() / max(ratios.min(), 1e-9))
            t_fit, fit_err, s = np.zeros(3), None, s_cluster
            if args.scale_mode == "image":
                s, t_fit, fit_err = refine_placement(Vr_c, s_cluster, center_world, size, ms.solvers)
                print(f"  scale: cluster-box {s_cluster:.4f} -> fitted to real masks {s:.4f}; "
                      f"moved {np.round(t_fit, 3)}; box fit error {100 * fit_err:.1f}% of object size in image")
                ratio = s / s_cluster
                if ratio > args.scale_ratio[1] and not args.accept_low:
                    print(f"  REJECTED: fitted size is {ratio:.1f}x the measured 3D cluster (a real object cannot be "
                          f"bigger than its own cluster) -- views and 3D points disagree; keeping the original mesh. "
                          f"(--accept-low to keep)")
                    fail_count += 1
                    continue
                if ratio < args.scale_ratio[0]:
                    print(f"  NOTE: fitted size is only {ratio:.2f}x the 3D cluster -> the cluster is inflated "
                          f"(swallowed floor/neighbours); the image fit is trusted")
                if fit_err > args.max_fit_error and not args.accept_low:
                    print(f"  REJECTED: mesh cannot be fitted consistently to all views "
                          f"(box error {100 * fit_err:.1f}% > {100 * args.max_fit_error:.0f}%); "
                          f"keeping the original mesh. (--accept-low to keep anyway)")
                    fail_count += 1
                    continue
            vs = Vr_c * s + c_loc + t_fit
            print(f"  final size {np.round(vs.max(0) - vs.min(0), 3)} vs cluster-measured {np.round(size, 3)} "
                  f"(cluster axis-ratio spread x{spread:.2f})")

            photo_cov = None
            if not args.no_photo_colors:
                view_data = [{"rgb": cv2.cvtColor(v["img"], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0,
                              "mask": v["mask"], "c2w": v["c2w"], "cam": v["cam"]} for v in vds]
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
                                        "scale_cluster": round(s_cluster, 5), "per_view_iou": per_view_iou,
                                        "placement_offset": [round(float(x), 4) for x in t_fit],
                                        "box_fit_error": None if fit_err is None else round(fit_err, 4),
                                        "scale_spread": round(spread, 3), "low_confidence": bool(low_conf),
                                        "crop_touches_border": bool(fc["touches"]),
                                        "photo_color_coverage": None if photo_cov is None else round(photo_cov, 3),
                                        "source_frame": fc["crop_frame"],
                                        "color_frames": [v["frame"] for v in vds]}
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
    print(f"Replaced: {ok_count}   failed/skipped/rejected: {fail_count}")
    print(f"Manifest updated: {manifest_path}")
    print("Check the *_align.png images in output\\generated (rows = views; left=mask, middle=render, right=overlap).")
    print("Next: python replace_splats.py --base-dir <capture_dir>   (removes the old splats)")


if __name__ == "__main__":
    main()