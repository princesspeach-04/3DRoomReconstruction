"""
add_capture.py

Add-on to run_pipeline.py for growing an existing capture with a new
video, WITHOUT deleting or re-processing what's already reconstructed.

Changes from the previous version:
  - FIXED: previously created a brand new camera per session
    (--ImageReader.single_camera 1 always makes a new camera entry).
    Now reads the existing model's camera ID and passes it via
    --ImageReader.existing_camera_id so new images share the SAME
    camera/intrinsics as the original capture (assuming same physical
    device -- correct almost all the time for this workflow).
  - Matching switched to vocab_tree_matcher (falls back to exhaustive
    for small totals). This is NOT sequential_matcher on purpose: a
    separately-recorded session isn't guaranteed to be a continuation
    of the original walk, so order-independent retrieval matching is
    the correct tool here, unlike in run_pipeline.py's initial video.
  - Bundle adjustment is now SKIPPED by default. Running a full global
    BA after every single incremental add doesn't scale -- it re-solves
    the entire combined model each time, not just the new part. Pass
    --finalize to run it once you're done adding for this session
    (run_pipeline.py's auto-extend loop does this automatically, once,
    after the whole loop finishes).
  - Callable via CLI now: `python add_capture.py --video path\to.mp4 [--finalize]`

Still does NOT do incremental gaussian splat training -- that's always a
full re-run over whatever images exist at that point.
"""
import os
import sys
import argparse
import subprocess
import struct
import time
import shutil
import collections
import cv2

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR_DEFAULT = r"C:\Users\Arni\UnityProjects\TestCapture"
COLMAP_EXE      = r"C:\Users\Arni\COLMAP\bin\colmap.exe"
TARGET_SAMPLE_FPS = 2.5   # match run_pipeline.py's sampling rate
# Leave blank to let COLMAP auto-download/cache the correct (Faiss-format)
# vocab tree itself on first use -- recent COLMAP versions do this. Only
# set this if your COLMAP is old enough to need an explicit path, and
# make sure whatever .bin you point at matches your COLMAP version (the
# older FLANN-format demuc.de file will crash any COLMAP built after the
# May 2025 Faiss switch).
VOCAB_TREE_PATH = ""
# ============================================================

def header(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)

def run(cmd, check=True):
    print(f"\n>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        print(f"\nERROR: command failed with code {result.returncode}")
        sys.exit(1)
    return result

def get_existing_camera_id(model_path):
    """Read cameras.bin and return the first camera's ID, so new images
    can be registered under the SAME camera instead of creating a new one."""
    path = os.path.join(model_path, "cameras.bin")
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        if num_cameras == 0:
            return None
        cam_id = struct.unpack("<i", f.read(4))[0]
        return cam_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to the new video to add")
    parser.add_argument("--base-dir", default=BASE_DIR_DEFAULT,
                         help="Capture folder -- MUST match whatever run_pipeline.py "
                              "used for this capture (it passes this automatically "
                              "when calling add_capture.py itself).")
    parser.add_argument("--finalize", action="store_true",
                         help="Run the (slow) full global bundle adjustment. "
                              "Only do this on the last add of a session.")
    parser.add_argument("--frame-offset", type=int, default=0,
                         help="Shift which frames get sampled, e.g. pass half of "
                              "the frame interval to pull the IN-BETWEEN frames "
                              "when re-running on the SAME video. Using offset 0 "
                              "twice on the same video re-extracts identical "
                              "frames -- pure waste, adds no new coverage.")
    args = parser.parse_args()

    global IMAGES_DIR, DB_PATH, SPARSE_DIR, OLD_MODEL
    BASE_DIR = args.base_dir
    IMAGES_DIR = os.path.join(BASE_DIR, "images")
    DB_PATH    = os.path.join(BASE_DIR, "database.db")
    SPARSE_DIR = os.path.join(BASE_DIR, "sparse")
    OLD_MODEL  = os.path.join(SPARSE_DIR, "0")

    if not os.path.exists(OLD_MODEL):
        print(f"ERROR: no existing model at {OLD_MODEL}.")
        print("Run run_pipeline.py first to create the initial reconstruction.")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"ERROR: new video not found at {args.video}")
        sys.exit(1)

    existing_camera_id = get_existing_camera_id(OLD_MODEL)
    if existing_camera_id is None:
        print("ERROR: could not find an existing camera in the model.")
        sys.exit(1)
    print(f"Reusing existing camera ID {existing_camera_id} for new images.")

    # ============================================================
    # STEP 1: extract new frames with collision-proof names
    # ============================================================
    header("STEP 1: Extracting frames from new video")

    session_tag = f"s{int(time.time())}"
    cap = cv2.VideoCapture(args.video, cv2.CAP_FFMPEG)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(fps / TARGET_SAMPLE_FPS))
    print(f"Video: {total_frames} frames at {fps:.1f} FPS")
    print(f"Targeting ~{TARGET_SAMPLE_FPS} extracted frames/sec -> every {frame_interval}th frame")

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if total_frames > 0 and count < total_frames * 0.9:
                print(f"WARNING: decoding stopped early at frame {count}/{total_frames}")
            break
        if count % frame_interval == (args.frame_offset % frame_interval):
            filename = f"frame_{session_tag}_{saved:04d}.jpg"
            filepath = os.path.join(IMAGES_DIR, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        count += 1
    cap.release()
    print(f"Extracted {saved} new frames (tagged '{session_tag}', old frames untouched)")

    if saved < 5:
        print("ERROR: too few new frames extracted, aborting before touching the model.")
        sys.exit(1)

    # ============================================================
    # STEP 2: feature extraction, reusing the existing camera
    # ============================================================
    header("STEP 2: Feature extraction on new images")
    run([COLMAP_EXE, "feature_extractor",
         "--database_path", DB_PATH,
         "--image_path", IMAGES_DIR,
         "--ImageReader.existing_camera_id", str(existing_camera_id)])

    # ============================================================
    # STEP 3: match new images against old + new (order-independent)
    # ============================================================
    header("STEP 3: Matching new images against existing set")
    cmd = [COLMAP_EXE, "vocab_tree_matcher", "--database_path", DB_PATH]
    if VOCAB_TREE_PATH:
        cmd += ["--VocabTreeMatching.vocab_tree_path", VOCAB_TREE_PATH]
    result = run(cmd, check=False)
    if result.returncode != 0:
        print("\nvocab_tree_matcher failed (old COLMAP without auto-download, or a")
        print("stale/incompatible vocab tree file). Falling back to exhaustive_matcher")
        print("-- fine for small total image counts, but you'll want to sort out the")
        print("vocab tree once your total capture grows past a couple hundred images.")
        run([COLMAP_EXE, "exhaustive_matcher", "--database_path", DB_PATH])

    # ============================================================
    # STEP 4: register new images into the EXISTING model
    # ============================================================
    header("STEP 4: Registering new images into existing model")
    NEW_MODEL = os.path.join(SPARSE_DIR, "1")
    if os.path.exists(NEW_MODEL):
        shutil.rmtree(NEW_MODEL)
    os.makedirs(NEW_MODEL)

    run([COLMAP_EXE, "image_registrator",
         "--database_path", DB_PATH,
         "--input_path", OLD_MODEL,
         "--output_path", NEW_MODEL])

    header("STEP 5: Triangulating new points")
    run([COLMAP_EXE, "point_triangulator",
         "--database_path", DB_PATH,
         "--image_path", IMAGES_DIR,
         "--input_path", NEW_MODEL,
         "--output_path", NEW_MODEL])

    if args.finalize:
        header("STEP 6: Bundle-adjusting combined model (finalize)")
        run([COLMAP_EXE, "bundle_adjuster",
             "--input_path", NEW_MODEL,
             "--output_path", NEW_MODEL,
             "--BundleAdjustment.max_num_iterations", "50"])
    else:
        print("\nSkipping global bundle adjustment for this add (pass --finalize")
        print("on the last add of a session to run it once, on the combined model).")

    # ============================================================
    # STEP 7: promote new model to be the "current" model
    # ============================================================
    header("STEP 7: Promoting new model")
    session_backup = os.path.join(SPARSE_DIR, f"0_backup_{session_tag}")
    shutil.move(OLD_MODEL, session_backup)
    shutil.move(NEW_MODEL, OLD_MODEL)
    print(f"Old model backed up to: {session_backup}")
    print("sparse/0 now contains the combined reconstruction.")

    result = subprocess.run([COLMAP_EXE, "model_analyzer", "--path", OLD_MODEL],
                             capture_output=True, text=True)
    print(result.stdout + result.stderr)

    print("\n" + "=" * 60)
    print("  DONE. Next steps:")
    print("  1. If more footage is queued, run add_capture.py again")
    print("     (only pass --finalize on the LAST one)")
    print("  2. Re-run compute_transform.py")
    print("  3. Re-run simple_trainer.py over the FULL images/ folder")
    print("  4. Re-run your YOLO detection step + detect_objects.py")
    print("=" * 60)


if __name__ == "__main__":
    main()