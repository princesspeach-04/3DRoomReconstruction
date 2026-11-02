"""
colmap_io.py  -- shared COLMAP binary readers (put next to your other scripts).

Fixes the old camera bug: the previous readers always read 4 doubles
(fx, fy, cx, cy). That is only right for the PINHOLE model. Your COLMAP run
uses the default SIMPLE_RADIAL model whose params are (f, cx, cy, k), so the old
code got fy = cx and cx = cy -> every projection (view-count filter, mask
voting) was shifted. This reader looks at the model id and returns correct
fx, fy, cx, cy.
(Radial distortion k is ignored: it is small, and gsplat trains on undistorted images.)
"""
import struct
import collections

Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name"])
Camera = collections.namedtuple("Camera", ["id", "model", "fx", "fy", "cx", "cy", "width", "height"])

# model_id: (name, n_params, layout)  layout "f"=single focal, "fxfy"=two focals
_MODELS = {
    0: ("SIMPLE_PINHOLE", 3, "f"), 1: ("PINHOLE", 4, "fxfy"), 2: ("SIMPLE_RADIAL", 4, "f"),
    3: ("RADIAL", 5, "f"), 4: ("OPENCV", 8, "fxfy"), 5: ("OPENCV_FISHEYE", 8, "fxfy"),
    6: ("FULL_OPENCV", 12, "fxfy"), 7: ("FOV", 5, "fxfy"), 8: ("SIMPLE_RADIAL_FISHEYE", 4, "f"),
    9: ("RADIAL_FISHEYE", 5, "f"), 10: ("THIN_PRISM_FISHEYE", 12, "fxfy"),
}


def read_cameras_binary(path):
    """Returns {camera_id: Camera}."""
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, mid = struct.unpack("<ii", f.read(8))
            w, h = struct.unpack("<QQ", f.read(16))
            if mid not in _MODELS:
                raise ValueError(f"Unknown COLMAP camera model id {mid}")
            name, npar, layout = _MODELS[mid]
            p = struct.unpack(f"<{npar}d", f.read(8 * npar))
            if layout == "f":
                fx = fy = p[0]; cx, cy = p[1], p[2]
            else:
                fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            cams[cid] = Camera(cid, name, fx, fy, cx, cy, w, h)
    return cams


def read_images_binary(path):
    """Returns {image_name: Image}."""
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            img_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = bytearray()
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            npts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * npts)  # x, y, point3D_id per 2D point
            images[name.decode("utf-8")] = Image(img_id, qvec, tvec, cam_id, name.decode("utf-8"))
    return images


def qvec_to_rotmat(q):
    w, x, y, z = q
    import numpy as np
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ])