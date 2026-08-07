"""
compute_transform.py

WHY THIS EXISTS:
gsplat's simple_trainer.py loads your COLMAP reconstruction with
Parser(..., normalize=True). That silently rotates, recenters, and
rescales your entire scene before training ever starts (see
gsplat/examples/datasets/normalize.py). The .ply you get out the other
end lives in THIS transformed space -- not in raw COLMAP space.

Your old scripts computed object positions from raw COLMAP data
(images.bin / points3D.bin) and/or projected into images using raw
COLMAP camera poses, while clustering points from the ALREADY-NORMALIZED
.ply. Mixing those two spaces is why the chair ended up in the wrong
place and got mislabeled.

This script recomputes the exact same transform gsplat applied
(same formulas, same order of operations, copied from gsplat's own
normalize.py) purely from your COLMAP sparse reconstruction, and saves
it to disk. detect_objects.py then uses it to move raw COLMAP camera
poses into the trained .ply's coordinate space before doing anything
else.

Run this once per capture, any time after COLMAP's mapper step
(STEP 3C in run_pipeline.py) has produced sparse/0/.
"""
import os
import json
import struct
import argparse
import collections
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", default=r"C:\Users\Arni\UnityProjects\TestCapture",
                     help="MUST match whatever run_pipeline.py used for this capture "
                          "(it auto-derives <video_name>_Capture unless you passed "
                          "--base-dir explicitly there too).")
args = parser.parse_args()

BASE_DIR   = args.base_dir
SPARSE_DIR = os.path.join(BASE_DIR, "sparse", "0")
OUT_FILE   = os.path.join(BASE_DIR, "output", "normalization.json")
# ============================================================

# ---- hand-rolled COLMAP binary readers (no pycolmap dependency --
# pycolmap's Python API has proven inconsistent across installs/versions;
# this is the same approach already used in detect_objects.py) ----

Image = collections.namedtuple("Image", ["id", "qvec", "tvec"])

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            struct.unpack("<i", f.read(4))  # camera_id, unused here
            while f.read(1) != b"\x00":     # skip null-terminated name
                pass
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(16 * n_pts)  # xys, unused here
            f.read(8 * n_pts)   # point3D_ids, unused here
            images[img_id] = Image(img_id, qvec, tvec)
    return images

def read_points3D_binary(path):
    xyzs = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            struct.unpack("<Q", f.read(8))       # point id, unused here
            xyz = struct.unpack("<3d", f.read(24))
            f.read(3)                            # rgb, unused here
            struct.unpack("<d", f.read(8))        # error, unused here
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(4 * track_length)  # image_ids, unused here
            f.read(4 * track_length)  # point2D_idxs, unused here
            xyzs.append(xyz)
    return np.array(xyzs, dtype=np.float64)

def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ])

# ---- copied verbatim (translated 1:1) from gsplat/examples/datasets/normalize.py ----

def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    if center_method == "focus":
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale
    return transform


def align_principal_axes(point_cloud):
    centroid = np.median(point_cloud, axis=0)
    translated = point_cloud - centroid
    cov = np.cov(translated, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    rotation_matrix = eigenvectors.T
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid
    return transform


def transform_points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix, camtoworlds):
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


# ---- main ----

print(f"Using BASE_DIR: {BASE_DIR}")
if not os.path.exists(SPARSE_DIR):
    raise FileNotFoundError(
        f"No sparse model at {SPARSE_DIR}. Either run_pipeline.py hasn't finished "
        f"the COLMAP step yet, or --base-dir doesn't match the folder it actually used."
    )
IMAGES_BIN = os.path.join(SPARSE_DIR, "images.bin")
POINTS_BIN = os.path.join(SPARSE_DIR, "points3D.bin")
assert os.path.exists(IMAGES_BIN), f"Missing {IMAGES_BIN}"
assert os.path.exists(POINTS_BIN), f"Missing {POINTS_BIN}"

print("Loading COLMAP reconstruction...")
images = read_images_binary(IMAGES_BIN)
points = read_points3D_binary(POINTS_BIN)

camtoworlds = []
for img_id in sorted(images.keys()):
    img = images[img_id]
    R = qvec_to_rotmat(img.qvec)
    t = np.array(img.tvec)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    camtoworlds.append(np.linalg.inv(w2c))
camtoworlds = np.stack(camtoworlds, axis=0)

print(f"{len(camtoworlds)} registered cameras, {len(points)} 3D points")

assert len(camtoworlds) >= 2, "Need at least 2 registered cameras to compute normalization."
assert len(points) >= 10, (
    f"Only {len(points)} 3D points in the sparse model -- too few for a stable "
    f"normalization. Reconstruction quality is likely too poor; check registered "
    f"image count and reconsider the capture before proceeding."
)

T1 = similarity_from_cameras(camtoworlds)
camtoworlds = transform_cameras(T1, camtoworlds)
points_t = transform_points(T1, points)

T2 = align_principal_axes(points_t)
camtoworlds = transform_cameras(T2, camtoworlds)
points_t = transform_points(T2, points_t)

transform = T2 @ T1
assert np.all(np.isfinite(transform)), (
    "Computed transform contains NaN/Inf -- likely degenerate camera geometry "
    "(e.g. all cameras nearly collinear or coincident). Check the reconstruction "
    "in a COLMAP viewer before trusting this capture."
)

# same "upside down" fix gsplat applies
if np.median(points_t[:, 2]) > np.mean(points_t[:, 2]):
    T3 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    transform = T3 @ transform
    print("Applied upside-down correction (T3), matching gsplat's own check.")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump({"transform": transform.tolist()}, f, indent=2)

print(f"\nSaved normalization transform to:\n  {OUT_FILE}")
print("detect_objects.py will use this to move COLMAP camera poses into")
print("the same coordinate space as your trained .ply.")