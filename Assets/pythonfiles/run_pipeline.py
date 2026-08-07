import os
import sys
import shutil
import argparse
import subprocess
import cv2

# ============================================================
# CLI ARGS -- run with any video: `python run_pipeline.py --video path\to\any.mp4`
# ============================================================
parser = argparse.ArgumentParser(description="Video -> COLMAP -> Gaussian Splat pipeline")
parser.add_argument("--video", default=r"C:\Users\Arni\UnityProjects\video.mp4",
                     help="Path to the input video. Can be any video.")
parser.add_argument("--base-dir", default=None,
                     help="Capture/output folder. Defaults to <video_name>_Capture "
                          "next to the video itself, so different videos don't "
                          "collide or overwrite each other.")
parser.add_argument("--target-registered", type=int, default=80,
                     help="Auto-extend loop stops once registered images reach this.")
cli_args = parser.parse_args()

VIDEO_PATH = cli_args.video
if cli_args.base_dir:
    BASE_DIR = cli_args.base_dir
else:
    video_stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    BASE_DIR = os.path.join(os.path.dirname(VIDEO_PATH), f"{video_stem}_Capture")
TARGET_REGISTERED = cli_args.target_registered

# ============================================================
# CONFIGURATION — installation-level paths, edit if these change
# ============================================================
COLMAP_EXE         = r"C:\Users\Arni\COLMAP\bin\colmap.exe"
GSPLAT_DIR         = r"C:\Users\Arni\UnityProjects\gsplat"
VENV_PYTHON        = r"C:\Users\Arni\UnityProjects\venv\Scripts\python.exe"

TARGET_SAMPLE_FPS  = 6   # extracted frames per second of video -- 2-3 is the
                           # sweet spot for photogrammetry/splatting; this
                           # replaces a fixed FRAME_INTERVAL so it self-adjusts
                           # to whatever the video's actual FPS is
MAX_STEPS          = 30000  # gaussian splatting training steps

MIN_FRAMES         = 20    # below this, ask before continuing at all
# Auto-extend reuses THIS SAME VIDEO_PATH with shifting --frame-offset
# values (see add_capture.py) instead of looking for new files -- each
# offset pulls a distinct set of in-between frames, not duplicates.
# ============================================================

IMAGES_DIR  = os.path.join(BASE_DIR, "images")
DB_PATH     = os.path.join(BASE_DIR, "database.db")
SPARSE_DIR  = os.path.join(BASE_DIR, "sparse")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
TRAINER     = os.path.join(GSPLAT_DIR, "examples", "simple_trainer.py")
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

def header(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)

def run(cmd, cwd=None, check=True):
    print(f"\n>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        print(f"\nERROR: command failed with code {result.returncode}")
        print("Pipeline stopped. Fix the error above and rerun.")
        sys.exit(1)
    return result

def get_registered_count(model_path):
    result = subprocess.run(
        [COLMAP_EXE, "model_analyzer", "--path", model_path],
        capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    registered = 0
    for line in output.split('\n'):
        if 'Registered images:' in line:
            try:
                # split on the label itself, not a bare ':' -- the raw
                # COLMAP log line has a timestamp like "07:49:18.680615"
                # BEFORE "Registered images:", so a naive split(':')[1]
                # grabs a fragment of the timestamp instead of the count
                registered = int(line.split('Registered images:')[-1].strip())
            except Exception:
                pass
    return registered, output

def spread_offsets(n):
    """Return a permutation of range(n) (bit-reversal / Van der Corput order)
    so each successive offset fills the biggest remaining gap between
    previously-sampled frames, instead of clustering samples together."""
    if n <= 1:
        return [0] if n == 1 else []
    bits = max(1, (n - 1).bit_length())
    size = 1 << bits
    seq, seen = [], set()
    for i in range(size):
        r = int(format(i, f'0{bits}b')[::-1], 2)
        if r < n and r not in seen:
            seq.append(r)
            seen.add(r)
    return seq

# ============================================================
# STEP 1: Extract frames from video
# ============================================================
header("STEP 1: Extracting frames from video")

if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: Video not found at {VIDEO_PATH}")
    print("Place your video file at that path and rerun.")
    sys.exit(1)

if os.path.exists(IMAGES_DIR):
    shutil.rmtree(IMAGES_DIR)
os.makedirs(IMAGES_DIR)

cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
FRAME_INTERVAL = max(1, round(fps / TARGET_SAMPLE_FPS))
print(f"Video: {total_frames} frames at {fps:.1f} FPS = {total_frames/fps:.1f} seconds")
print(f"Targeting ~{TARGET_SAMPLE_FPS} extracted frames/sec -> every {FRAME_INTERVAL}th frame")

count = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print(f"Decoding stopped at frame {count} (container reported {total_frames} total)")
        if total_frames > 0 and count < total_frames * 0.9:
            print("WARNING: stopped well short of the reported frame count -- likely a")
            print("decoder issue (e.g. GPU contention from a prior run), not end of video.")
        break
    if count % FRAME_INTERVAL == 0:
        filename = os.path.join(IMAGES_DIR, f"frame_{saved:04d}.jpg")
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    count += 1
cap.release()
print(f"Extracted {saved} frames")

if saved < MIN_FRAMES:
    print(f"\nWARNING: only {saved} frames extracted (recommended minimum {MIN_FRAMES}).")
    answer = input("Continue anyway? (y/n): ")
    if answer.lower() != 'y':
        print("Pipeline stopped. Retake video and rerun.")
        sys.exit(0)

# ============================================================
# STEP 2: Clear old COLMAP data
# ============================================================
header("STEP 2: Clearing old COLMAP data")

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print(f"Deleted: {DB_PATH}")

for path in [SPARSE_DIR, OUTPUT_DIR]:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted: {path}")
    os.makedirs(path)
    print(f"Created: {path}")

# ============================================================
# STEP 3: COLMAP
# ============================================================
header("STEP 3A: COLMAP Feature Extraction")
run([COLMAP_EXE, "feature_extractor",
     "--database_path", DB_PATH,
     "--image_path", IMAGES_DIR,
     "--ImageReader.single_camera", "1"])

header("STEP 3B: COLMAP Feature Matching")
# sequential_matcher instead of exhaustive_matcher: these frames come from
# ONE continuous video walk, so they're genuinely sequential -- this only
# matches nearby-in-time frames instead of all O(n^2) pairs. Do NOT use
# sequential_matcher in add_capture.py though -- a separately-recorded
# session isn't guaranteed to be a continuation of this walk.
run([COLMAP_EXE, "sequential_matcher",
     "--database_path", DB_PATH])

header("STEP 3C: COLMAP Sparse Reconstruction")
run([COLMAP_EXE, "mapper",
     "--database_path", DB_PATH,
     "--image_path", IMAGES_DIR,
     "--output_path", SPARSE_DIR])

header("Checking reconstruction quality...")
registered, output = get_registered_count(os.path.join(SPARSE_DIR, "0"))
print(output)
print(f"\nRegistered images: {registered} / {saved}")

if registered < 15:
    print("\nWARNING: Very few images registered.")
    print("Reconstruction quality will be poor.")
    print("Consider retaking video more slowly with better lighting.")
    answer = input("Continue anyway? (y/n): ")
    if answer.lower() != 'y':
        print("Pipeline stopped. Retake video and rerun.")
        sys.exit(0)

# ============================================================
# STEP 3D: Auto-extend by reusing the SAME video at shifting offsets
# ============================================================
if registered < TARGET_REGISTERED:
    header(f"STEP 3D: {registered}/{TARGET_REGISTERED} registered -- resampling {os.path.basename(VIDEO_PATH)}")

    # offset 0 is already used (that's the original Step 1 extraction),
    # so skip it here and only try the remaining distinct phases
    offsets_to_try = [o for o in spread_offsets(FRAME_INTERVAL) if o != 0]
    print(f"Frame interval is {FRAME_INTERVAL} -> up to {len(offsets_to_try)} more distinct")
    print(f"offsets available from this same video before frames start repeating.")

    added_any = False
    idx = 0
    while registered < TARGET_REGISTERED and idx < len(offsets_to_try):
        offset = offsets_to_try[idx]
        idx += 1
        print(f"\nRegistered {registered}/{TARGET_REGISTERED} -- reusing video at frame-offset {offset}")
        run([VENV_PYTHON, os.path.join(SCRIPT_DIR, "add_capture.py"),
             "--video", VIDEO_PATH,
             "--base-dir", BASE_DIR,
             "--frame-offset", str(offset)])
        added_any = True
        registered, _ = get_registered_count(os.path.join(SPARSE_DIR, "0"))
        print(f"Now registered: {registered}/{TARGET_REGISTERED}")

    if registered < TARGET_REGISTERED and idx >= len(offsets_to_try):
        print(f"\nExhausted every distinct frame offset obtainable from this video")
        print(f"at {TARGET_SAMPLE_FPS} fps sampling -- every frame has now been used at")
        print(f"least once. Proceeding with {registered} registered images.")
        print(f"To get more from the same footage, lower TARGET_SAMPLE_FPS (more, denser")
        print(f"offsets to draw from); to go further than that, you need new footage.")

    if added_any:
        header("Finalizing combined model (single global bundle adjustment)")
        run([COLMAP_EXE, "bundle_adjuster",
             "--input_path", os.path.join(SPARSE_DIR, "0"),
             "--output_path", os.path.join(SPARSE_DIR, "0"),
             "--BundleAdjustment.max_num_iterations", "50"])
else:
    print(f"\nAlready at {registered}/{TARGET_REGISTERED} target, skipping auto-extend.")

# ============================================================
# STEP 4: Gaussian Splatting Training
# ============================================================
header("STEP 4: Training Gaussian Splat")
print("Training time scales with total image count -- expect longer than")
print("10-20 min if additional footage was added above.")
print(f"Training for {MAX_STEPS} steps...")

run([VENV_PYTHON, TRAINER,
     "default",
     "--data_dir", BASE_DIR,
     "--data_factor", "1",
     "--result_dir", OUTPUT_DIR,
     "--max_steps", str(MAX_STEPS),
     "--save_ply",
     "--disable_viewer"],
    cwd=GSPLAT_DIR)

# ============================================================
# STEP 5: Verify output
# ============================================================
header("STEP 5: Verifying output")

ply_path = os.path.join(OUTPUT_DIR, "ply", f"point_cloud_{MAX_STEPS-1}.ply")
if os.path.exists(ply_path):
    size_mb = os.path.getsize(ply_path) / (1024 * 1024)
    print(f"\nSUCCESS! PLY file created: {ply_path}")
    print(f"File size: {size_mb:.1f} MB")
    print(f"\nNow load into Unity:")
    print(f"1. Tools → Gaussian Splats → Create GaussianSplatAsset")
    print(f"2. Input PLY: {ply_path}")
    print(f"3. Output Folder: Assets/SplatAssets")
    print(f"4. Click Create Asset")
    print(f"5. Assign to GaussianSplats GameObject → Press Play")
else:
    print(f"\nERROR: PLY file not found at expected path: {ply_path}")
    print("Training may have failed. Check output above for errors.")
    sys.exit(1)

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)