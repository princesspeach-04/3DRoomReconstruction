"""
run_postprocess.py  -- everything AFTER training + cleaning, in one command, with per-stage timing.

Stages (in order):
  yolo      run_yolo_segmentation.py
  objects   build_object_meshes.py
  classify  classify_treatment.py
  replace   generate_replacement.py   (automatic: guards skip junk, bad shapes are rejected)
  splats    replace_splats.py
  collider  build_collider_voxel.py   (from the *_clean_replaced PLY if any object was replaced)

Usage:
  python run_postprocess.py --base-dir %BASE_DIR% --triposr-dir %TRIPOSR%
  python run_postprocess.py --base-dir %BASE_DIR% --triposr-dir %TRIPOSR% --from-stage replace
  python run_postprocess.py --base-dir %BASE_DIR% --triposr-dir %TRIPOSR% --skip yolo objects
Stops at the first failing stage. Extra flags for generate_replacement.py can be appended
after --  e.g.  ... -- --min-iou 0.4 --max-color-views 6
"""
import os
import sys
import json
import time
import argparse
import subprocess

STAGES = ["yolo", "objects", "classify", "replace", "splats", "collider"]

ap = argparse.ArgumentParser()
ap.add_argument("--base-dir", required=True)
ap.add_argument("--triposr-dir", required=True)
ap.add_argument("--max-steps", type=int, default=30000)
ap.add_argument("--yolo-weights", default="yolov8n-seg.pt",
                help="e.g. yolov8m-seg.pt or yolov8x-seg.pt: bigger = finds more/dark objects (auto-downloads)")
ap.add_argument("--yolo-conf", type=float, default=0.25, help="lower (0.15) finds weaker detections")
ap.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
ap.add_argument("--skip", nargs="+", choices=STAGES, default=[])
ap.add_argument("extra", nargs="*", help="extra args for generate_replacement.py (put after --)")
args = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable
base = os.path.abspath(args.base_dir)
steps = str(args.max_steps)


def script(name):
    return os.path.join(here, name)


def replaced_any():
    try:
        with open(os.path.join(base, "segmentation", "objects_3d.json")) as f:
            return any(o.get("mesh_source") == "generative_replace_triposr" for o in json.load(f))
    except Exception:
        return False


def cmd_for(stage):
    if stage == "yolo":
        return [py, script("run_yolo_segmentation.py"), "--base-dir", base,
                "--weights", args.yolo_weights, "--conf", str(args.yolo_conf)]
    if stage == "objects":
        return [py, script("build_object_meshes.py"), "--base-dir", base, "--max-steps", steps,
                "--dbscan-eps", "0.05", "--min-cluster-size", "500"]
    if stage == "classify":
        return [py, script("classify_treatment.py"), "--base-dir", base]
    if stage == "replace":
        return [py, script("generate_replacement.py"), "--base-dir", base,
                "--triposr-dir", os.path.abspath(args.triposr_dir)] + args.extra
    if stage == "splats":
        return [py, script("replace_splats.py"), "--base-dir", base, "--max-steps", steps]
    if stage == "collider":
        c = [py, script("build_collider_voxel.py"), "--capture-dir", base,
             "--out", os.path.join(base, "output", "room_mesh.obj")]
        if replaced_any():
            c += ["--ply-suffix", "_clean_replaced"]
        return c


timings = []
start = STAGES.index(args.from_stage)
t_all = time.time()
for stage in STAGES[start:]:
    if stage in args.skip:
        print(f"\n--- skipping {stage}")
        continue
    if stage == "splats" and not replaced_any():
        print("\n--- splats: no object was replaced, nothing to cut out")
        continue
    cmd = cmd_for(stage)
    print("\n" + "=" * 60)
    print(f"  STAGE {stage}")
    print("=" * 60)
    print(">>", " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = time.time() - t0
    timings.append((stage, dt))
    if r.returncode != 0:
        print(f"\nSTAGE '{stage}' FAILED (exit {r.returncode}). Fix it, then rerun with --from-stage {stage}")
        break

print("\n" + "=" * 60)
print("  TIMINGS")
print("=" * 60)
for s, d in timings:
    print(f"  {s:10s} {d / 60:6.1f} min")
print(f"  {'total':10s} {(time.time() - t_all) / 60:6.1f} min")