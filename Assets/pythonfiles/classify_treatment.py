"""
classify_treatment.py

Reads objects_3d.json (already has class, point_count, triangle_count,
bbox_aspect_ratio from build_object_meshes.py) and assigns each object a
"treatment":
  keep_reconstructed  - mesh is good enough as-is
  regularize_plane    - flat/regular objects (tables, floors) that should
                         be geometrically snapped rather than left noisy
  generative_replace   - small/complex/thin objects that alpha-shape
                         reconstruction genuinely can't resolve well

No AI call needed -- this is a deterministic rule table (class-based)
plus a quality-metric fallback (catches a normally-fine class that came
out badly in this specific capture).

Usage:
    python classify_treatment.py --base-dir <capture_dir>
"""
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", required=True)
parser.add_argument("--min-triangle-count", type=int, default=300,
                     help="Below this, a mesh is considered too sparse to "
                          "trust regardless of class -- flagged for "
                          "generative replacement.")
args = parser.parse_args()

BASE_DIR = args.base_dir
SEG_DIR = os.path.join(BASE_DIR, "segmentation")
manifest_path = os.path.join(SEG_DIR, "objects_3d.json")

# Class -> default treatment. Extend this as you add more YOLO classes.
CLASS_TREATMENT = {
    "chair":  "keep_reconstructed",
    "table":  "regularize_plane",
    "floor":  "regularize_plane",
    "laptop": "generative_replace",
    "phone":  "generative_replace",
    "bottle": "generative_replace",
    "cup":    "generative_replace",
    "bag":    "generative_replace",
}

with open(manifest_path) as f:
    manifest = json.load(f)

counts = {"keep_reconstructed": 0, "regularize_plane": 0, "generative_replace": 0}

for obj in manifest:
    cls = obj["class"]
    default_treatment = CLASS_TREATMENT.get(cls, "keep_reconstructed")

    # quality-metric override: even a class that's usually fine gets
    # flagged if THIS specific instance came out too sparse to trust
    # (flat/support classes such as table/floor are exempt: a small fragment of a table is NOT a
    #  small object, and must never be sent to the object generator)
    if obj.get("triangle_count", 9999) < args.min_triangle_count and default_treatment != "regularize_plane":
        treatment = "generative_replace"
    else:
        treatment = default_treatment

    obj["treatment"] = treatment
    counts[treatment] += 1
    print(f"{obj['id']:15s} class={cls:10s} tris={obj.get('triangle_count','?'):5} "
          f"aspect={obj.get('bbox_aspect_ratio','?'):5} -> {treatment}")

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n{counts}")
print(f"Updated: {manifest_path}")