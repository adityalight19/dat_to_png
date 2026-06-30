import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_png(img, output_file, cmap="gray", vmin=0, vmax=1, transpose_image=True):
    """
    Save one 2D slice as PNG.
    transpose_image=True gives a seismic-style display orientation.
    """
    if transpose_image:
        img = img.T

    plt.imsave(
        output_file,
        img,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )


def dat_to_all_orientations(
    dat_path,
    out_root,
    shape=(343, 615, 1252),
    dtype=np.single,
    cmap="gray",
    vmin=0,
    vmax=1,
    transpose_image=True
):
    dat_path = Path(dat_path)
    out_root = Path(out_root)

    # Output folders
    inline_dir = out_root / "inline"
    crossline_dir = out_root / "crossline"
    timeline_dir = out_root / "timeline"

    inline_dir.mkdir(parents=True, exist_ok=True)
    crossline_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir.mkdir(parents=True, exist_ok=True)

    print("Reading DAT file from:")
    print(dat_path)

    data = np.fromfile(dat_path, dtype=dtype)

    expected_size = shape[0] * shape[1] * shape[2]

    print("File values:", data.size)
    print("Expected values:", expected_size)

    if data.size != expected_size:
        raise ValueError(
            f"Size mismatch. File has {data.size} values, "
            f"but shape {shape} needs {expected_size} values."
        )

    # Shape is assumed as:
    # shape = (inline, crossline, time)
    volume = data.reshape(shape)

    print("Volume shape:", volume.shape)

    n_inline, n_crossline, n_time = volume.shape

    # -----------------------
    # Inline images
    # Fixed inline index
    # Output: crossline vs time
    # -----------------------
    for i in range(n_inline):
        img = volume[i, :, :]
        output_file = inline_dir / f"inline_{i:03d}.png"
        save_png(img, output_file, cmap, vmin, vmax, transpose_image)

    print(f"Saved {n_inline} inline images in:")
    print(inline_dir)

    # -----------------------
    # Crossline images
    # Fixed crossline index
    # Output: inline vs time
    # -----------------------
    for i in range(n_crossline):
        img = volume[:, i, :]
        output_file = crossline_dir / f"crossline_{i:03d}.png"
        save_png(img, output_file, cmap, vmin, vmax, transpose_image)

    print(f"Saved {n_crossline} crossline images in:")
    print(crossline_dir)

    # -----------------------
    # Timeline / Time-slice images
    # Fixed time index
    # Output: inline vs crossline
    # -----------------------
    for i in range(n_time):
        img = volume[:, :, i]
        output_file = timeline_dir / f"timeline_{i:04d}.png"
        save_png(img, output_file, cmap, vmin, vmax, transpose_image=False)

    print(f"Saved {n_time} timeline images in:")
    print(timeline_dir)

    print("Done.")


# ----------------------------
# FILE LOCATION AND OUTPUT FOLDER
# ----------------------------

dat_to_all_orientations(
    dat_path="PATH",
    out_root="PATH",
    shape=(343, 615, 1252),
    dtype=np.single,
    cmap="gray",
    vmin=0,
    vmax=1,
    transpose_image=True
)
