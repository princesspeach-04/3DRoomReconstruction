"""
run_yolo_segmentation.py

Replaces run_yolo_detection.py. Instead of bounding boxes, this runs
YOLOv8-SEG and saves a per-instance PIXEL MASK for every detection.
Boxes only ever gave you a rectangular blob of 3D points when
backprojected -- that's the "box dropping from somewhere" problem.
Masks let build_object_meshes.py keep only the points that actually
lie on the object's silhouette in each frame.

Output:
  segmentation/detections_seg.json
      { "frame_0000.jpg": [ {"class": "chair", "confidence": 0.91,
                              "mask_key": "mask_0"}, ... ], ... }
  segmentation/masks/<frame_stem>.npz
      one compressed boolean array per instance, key "mask_0", "mask_1", ...
      shape (height, width), True where that instance covers the pixel

Usage:
    python run_yolo_segmentation.py --base-dir <capture_dir> --classes chair
"""
import os
import json
import argparse
import numpy as np
import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir", required=True,
                     help="MUST match whatever run_pipeline.py used for this capture.")
parser.add_argument("--weights", default="yolov8n-seg.pt",
                     help="Segmentation weights -- NOT the plain yolov8n.pt "
                          "used by the old box-based script. Auto-downloads "
                          "if not found locally.")
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--classes", nargs="+", default=["chair"],
                     help="Only keep these COCO classes -- filters noise and "
                          "keeps mask storage small. Add more later once "
                          "the chair-only pipeline is validated end to end.")
args = parser.parse_args()

BASE_DIR = args.base_dir
IMAGES_DIR = os.path.join(BASE_DIR, "images")
SEG_DIR = os.path.join(BASE_DIR, "segmentation")
MASKS_DIR = os.path.join(SEG_DIR, "masks")
OUT_FILE = os.path.join(SEG_DIR, "detections_seg.json")

print(f"Using BASE_DIR: {BASE_DIR}")
assert os.path.isdir(IMAGES_DIR), f"No images folder at {IMAGES_DIR}"
os.makedirs(MASKS_DIR, exist_ok=True)

image_files = sorted(
    f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
)
assert len(image_files) > 0, f"No images found in {IMAGES_DIR}"
print(f"Found {len(image_files)} images, filtering to classes: {args.classes}")

model = YOLO(args.weights)
class_filter = set(args.classes)

detections = {}
total_instances = 0

for i, fname in enumerate(image_files):
    fpath = os.path.join(IMAGES_DIR, fname)
    results = model(fpath, conf=args.conf, verbose=False)
    entries = []
    mask_arrays = {}

    for r in results:
        if r.masks is None:
            continue
        img_h, img_w = r.orig_shape
        # r.masks.data: (N, mask_h, mask_w) -- often a different (lower)
        # resolution than the source image, so resize each mask back up
        # to the ORIGINAL frame resolution before saving. Getting this
        # wrong silently misaligns every downstream pixel test.
        raw_masks = r.masks.data.cpu().numpy()
        for j, box in enumerate(r.boxes):
            cls_name = model.names[int(box.cls)]
            if cls_name not in class_filter:
                continue
            conf = float(box.conf)
            mask = raw_masks[j]
            if mask.shape != (img_h, img_w):
                mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask > 0.5

            key = f"mask_{len(entries)}"
            mask_arrays[key] = mask_bool
            entries.append({"class": cls_name, "confidence": conf, "mask_key": key})

    if entries:
        stem = os.path.splitext(fname)[0]
        np.savez_compressed(os.path.join(MASKS_DIR, f"{stem}.npz"), **mask_arrays)

    detections[fname] = entries
    total_instances += len(entries)
    if (i + 1) % 20 == 0 or i == len(image_files) - 1:
        print(f"  processed {i + 1}/{len(image_files)} images, "
              f"{total_instances} instances so far")

assert total_instances > 0, (
    f"Zero instances of {args.classes} found across every image. Either "
    f"--conf is too high, the class name doesn't match COCO's exact "
    f"naming, or the scene genuinely doesn't contain the target object."
)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w") as f:
    json.dump(detections, f, indent=2)

print(f"\nSaved {total_instances} instance masks across {len(image_files)} images to:")
print(f"  {OUT_FILE}")
print(f"  {MASKS_DIR}/")