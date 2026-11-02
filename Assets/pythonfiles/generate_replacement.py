"""
generate_replacement.py

For EVERY object flagged treatment=generative_replace in objects_3d.json
(no class hardcoding -- works for any object, any count of instances of
the same class, on any capture):

  1. Uses that object's own best_frame/best_mask_key (from
     build_object_meshes.py's per-instance matching) to crop exactly
     that instance -- not "some object of this class somewhere".
  2. Runs TripoSR locally on the crop.
  3. Searches the actual output directory for the generated mesh file
     rather than assuming an exact path, so it's robust to TripoSR
     version differences.
  4. Updates objects_3d.json's mesh_file to point at the generated mesh,
     keeping the original position/size from real reconstruction.
  5. Skips gracefully (never crashes the whole run) on any single
     object's failure -- logs it and moves to the next.

Usage:
    python generate_replacement.py --base-dir <capture_dir> --triposr-dir <path_to_TripoSR>
"""
import os
import sys
import json
import glob
import argparse
import subprocess
import numpy as np
from PIL import Image
# ---- add near the top of generate_replacement.py ----
import struct
import collections

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            struct.unpack("<i", f.read(4))
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            struct.unpack("<i", f.read(4))
            name_bytes = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(16 * n_pts)
            f.read(8 * n_pts)
            images[name_bytes.decode("utf-8")] = (qvec, tvec)
    return images

def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ])

def get_camera_position(fname, images, T):
    qvec, tvec = images[fname]
    R = qvec_to_rotmat(qvec)
    t = np.array(tvec)
    w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)
    c2w_t = T @ c2w
    scale = np.linalg.norm(c2w_t[:3, 0])
    c2w_t[:3, :3] /= scale
    return c2w_t[:3, 3]  # camera position in normalized scene space

def detect_up_axis(room_mesh_path):
    """Auto-detect which axis is 'vertical' from the room mesh's own shape --
    rooms are reliably wider/longer than they are tall, so the SMALLEST
    extent axis is up. Class-agnostic, capture-agnostic."""
    import trimesh
    room = trimesh.load(room_mesh_path, force="mesh")
    extent = room.bounds[1] - room.bounds[0]
    return int(np.argmin(extent))

def align_generated_mesh(gen_mesh, obj_position, cam_position, up_axis):
    """Rotate a freshly-generated mesh so its 'forward' faces back toward
    the camera that captured it, and its 'up' matches the room's real up
    axis. Fully class-agnostic -- uses only geometry, no object identity.

    CAVEAT: assumes TripoSR's canonical output convention is +Y up,
    +Z forward-facing-camera. Verify on your first result -- if the
    object comes out facing backward/sideways, flip CANON_FORWARD's
    sign below rather than re-deriving the whole thing."""
    CANON_UP = np.array([0, 1, 0])
    CANON_FORWARD = np.array([0, 0, 1])

    world_up = np.zeros(3); world_up[up_axis] = 1.0
    horiz_dir = cam_position - obj_position
    horiz_dir[up_axis] = 0
    norm = np.linalg.norm(horiz_dir)
    if norm < 1e-6:
        return gen_mesh  # camera directly above/below, no reliable heading
    horiz_dir /= norm

    def rotation_between(a, b):
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s < 1e-8:
            return np.eye(3)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    R_up = rotation_between(CANON_UP, world_up)
    fwd_rotated = R_up @ CANON_FORWARD
    fwd_rotated[up_axis] = 0
    fwd_rotated /= (np.linalg.norm(fwd_rotated) + 1e-8)
    R_yaw = rotation_between(fwd_rotated, horiz_dir)

    cam_pos = get_camera_position(best_frame, images, T)
    gen_mesh = align_generated_mesh(gen_mesh, np.array(obj["position"]), cam_pos, up_axis)

    gen_mesh.vertices = (R_yaw @ R_up @ gen_mesh.vertices.T).T
    return gen_mesh

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", required=True)
parser.add_argument("--triposr-dir", required=True)
parser.add_argument("--crop-padding", type=int, default=20)
parser.add_argument("--classes", nargs="+", default=None,
                     help="Optional: restrict to specific classes only "
                          "(e.g. --classes laptop). Default: process every "
                          "flagged object regardless of class.")
args = parser.parse_args()

BASE_DIR = args.base_dir
SEG_DIR = os.path.join(BASE_DIR, "segmentation")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
MASKS_DIR = os.path.join(SEG_DIR, "masks")
GEN_DIR = os.path.join(BASE_DIR, "output", "generated")
os.makedirs(GEN_DIR, exist_ok=True)

manifest_path = os.path.join(SEG_DIR, "objects_3d.json")
with open(manifest_path) as f:
    manifest = json.load(f)
images_bin = os.path.join(BASE_DIR, "sparse", "0", "images.bin")
with open(os.path.join(BASE_DIR, "output", "normalization.json")) as f:
    T = np.array(json.load(f)["transform"])
images = read_images_binary(images_bin)
up_axis = detect_up_axis(os.path.join(BASE_DIR, "output", "room_mesh.obj"))
print(f"Detected up-axis: {['X','Y','Z'][up_axis]}")

to_replace = [o for o in manifest if o.get("treatment") == "generative_replace"]
if args.classes:
    to_replace = [o for o in to_replace if o["class"] in args.classes]
print(f"{len(to_replace)} object(s) queued for generative replacement")

triposr_run = os.path.join(args.triposr_dir, "run.py")
succeeded, failed = 0, 0

for obj in to_replace:
    oid = obj["id"]
    best_frame = obj.get("best_frame")
    best_mask_key = obj.get("best_mask_key")

    if not best_frame or not best_mask_key:
        print(f"\n{oid}: no best_frame recorded (rerun build_object_meshes.py "
              f"with the per-instance matching addition), skipping")
        failed += 1
        continue

    print(f"\n{oid} (class={obj['class']}): using {best_frame} "
          f"(overlap={obj.get('best_frame_overlap','?')}, "
          f"conf={obj.get('best_frame_confidence','?')})")

    img_path = os.path.join(IMAGES_DIR, best_frame)
    stem = os.path.splitext(best_frame)[0]
    npz_path = os.path.join(MASKS_DIR, f"{stem}.npz")
    if not os.path.exists(img_path) or not os.path.exists(npz_path):
        print(f"  missing source frame or mask file, skipping")
        failed += 1
        continue

    masks = np.load(npz_path)
    mask = masks[best_mask_key]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        print(f"  empty mask, skipping")
        failed += 1
        continue

    x0 = max(int(xs.min()) - args.crop_padding, 0)
    x1 = int(xs.max()) + args.crop_padding
    y0 = max(int(ys.min()) - args.crop_padding, 0)
    y1 = int(ys.max()) + args.crop_padding

    img = Image.open(img_path)
    crop = img.crop((x0, y0, x1, y1))
    crop_path = os.path.join(GEN_DIR, f"{oid}_crop.png")
    crop.save(crop_path)
    print(f"  cropped ({crop.size}) -> {crop_path}")

    obj_out_dir = os.path.join(GEN_DIR, oid)
    os.makedirs(obj_out_dir, exist_ok=True)

    result = subprocess.run([
        sys.executable, triposr_run,
        crop_path,
        "--output-dir", obj_out_dir,
        "--model-save-format", "obj",
    ], cwd=args.triposr_dir)

    if result.returncode != 0:
        print(f"  TripoSR failed (exit {result.returncode}), keeping original mesh")
        failed += 1
        continue

    # ---- search for the generated mesh instead of assuming exact path ----
    candidates = glob.glob(os.path.join(obj_out_dir, "**", "mesh.obj"), recursive=True)
    if not candidates:
        print(f"  no mesh.obj found under {obj_out_dir}, keeping original mesh")
        failed += 1
        continue

    generated_obj = candidates[0]
    obj["mesh_file"] = generated_obj
    obj["mesh_source"] = "generative_replace_triposr"
    print(f"  mesh_file -> {generated_obj}")
    succeeded += 1

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n{'='*50}")
print(f"Succeeded: {succeeded}, Failed/skipped: {failed}")
print(f"Updated manifest: {manifest_path}")