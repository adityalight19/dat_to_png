import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import numpy as np
import tifffile


def normalize_to_uint16(arr, p_low=1, p_high=99):
    """
    Normalize seismic amplitude data to uint16 for TIFF viewing.
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(arr, [p_low, p_high])

    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint16)

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)
    arr = arr * 65535.0

    return arr.astype(np.uint16)


def rotate_array(arr, rotation_degrees):
    """
    Rotate a 2D image or every YX slice of a 3D volume.

    rotation_degrees can be 0, 90, 180, or 270.
    """
    if rotation_degrees == 0:
        return arr

    if rotation_degrees not in [90, 180, 270]:
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees.")

    k = rotation_degrees // 90

    if arr.ndim == 2:
        return np.rot90(arr, k=k)

    elif arr.ndim == 3:
        # Rotate each 2D slice in the last two dimensions: YX
        return np.rot90(arr, k=k, axes=(-2, -1))

    else:
        raise ValueError("Only 2D or 3D arrays can be rotated.")


def save_volume_tiff(arr, output_path, normalize=True, rotation_degrees=0):
    """
    Save 2D or 3D array as a TIFF.
    3D arrays are saved as multi-page TIFF stacks.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr_to_save = rotate_array(arr, rotation_degrees)

    if normalize:
        arr_to_save = normalize_to_uint16(arr_to_save)

    metadata = {"axes": "ZYX"} if arr_to_save.ndim == 3 else {"axes": "YX"}

    tifffile.imwrite(
        output_path,
        arr_to_save,
        photometric="minisblack",
        metadata=metadata,
    )


def save_slices(arr, output_dir, axis=0, normalize=True, rotation_degrees=0):
    """
    Save each 2D slice of a 3D volume as an individual TIFF.
    """
    if arr.ndim != 3:
        raise ValueError("Slice export requires a 3D array.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_slices = arr.shape[axis]

    for i in range(n_slices):
        if axis == 0:
            slc = arr[i, :, :]
        elif axis == 1:
            slc = arr[:, i, :]
        elif axis == 2:
            slc = arr[:, :, i]
        else:
            raise ValueError("Axis must be 0, 1, or 2.")

        slc = rotate_array(slc, rotation_degrees)

        if normalize:
            slc = normalize_to_uint16(slc)

        output_file = output_dir / f"slice_axis{axis}_{i:04d}.tif"
        tifffile.imwrite(output_file, slc, photometric="minisblack")


class NpyToTiffGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Seismic NPY to TIFF Converter")
        self.root.geometry("620x470")
        self.root.resizable(False, False)

        self.input_file = tk.StringVar()
        self.output_path = tk.StringVar()
        self.export_mode = tk.StringVar(value="stack")
        self.axis = tk.IntVar(value=0)
        self.normalize = tk.BooleanVar(value=True)
        self.rotation = tk.IntVar(value=0)

        self.create_widgets()

    def create_widgets(self):
        padding = {"padx": 12, "pady": 8}

        title = ttk.Label(
            self.root,
            text="Seismic .NPY to TIFF Converter",
            font=("Arial", 16, "bold"),
        )
        title.pack(pady=15)

        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=20)

        ttk.Label(frame, text="Input .npy file:").grid(row=0, column=0, sticky="w", **padding)

        input_entry = ttk.Entry(frame, textvariable=self.input_file, width=55)
        input_entry.grid(row=1, column=0, sticky="w", padx=12)

        ttk.Button(
            frame,
            text="Browse",
            command=self.select_input_file,
        ).grid(row=1, column=1, padx=8)

        ttk.Label(frame, text="Output TIFF file or folder:").grid(row=2, column=0, sticky="w", **padding)

        output_entry = ttk.Entry(frame, textvariable=self.output_path, width=55)
        output_entry.grid(row=3, column=0, sticky="w", padx=12)

        ttk.Button(
            frame,
            text="Browse",
            command=self.select_output_path,
        ).grid(row=3, column=1, padx=8)

        ttk.Label(frame, text="Export mode:").grid(row=4, column=0, sticky="w", **padding)

        mode_frame = ttk.Frame(frame)
        mode_frame.grid(row=5, column=0, sticky="w", padx=12)

        ttk.Radiobutton(
            mode_frame,
            text="Single multi-page TIFF stack",
            variable=self.export_mode,
            value="stack",
            command=self.update_output_label,
        ).pack(anchor="w")

        ttk.Radiobutton(
            mode_frame,
            text="Separate TIFF slices",
            variable=self.export_mode,
            value="slices",
            command=self.update_output_label,
        ).pack(anchor="w")

        ttk.Label(frame, text="Slice axis, only for separate slices:").grid(
            row=6, column=0, sticky="w", **padding
        )

        axis_combo = ttk.Combobox(
            frame,
            textvariable=self.axis,
            values=[0, 1, 2],
            width=10,
            state="readonly",
        )
        axis_combo.grid(row=7, column=0, sticky="w", padx=12)

        axis_help = ttk.Label(
            frame,
            text="Axis 0 = inline-style, Axis 1 = crossline-style, Axis 2 = time/depth slices",
            foreground="gray",
        )
        axis_help.grid(row=8, column=0, sticky="w", padx=12, pady=4)

        ttk.Label(frame, text="Rotate generated TIFF image:").grid(
            row=9, column=0, sticky="w", **padding
        )

        rotation_combo = ttk.Combobox(
            frame,
            textvariable=self.rotation,
            values=[0, 90, 180, 270],
            width=10,
            state="readonly",
        )
        rotation_combo.grid(row=10, column=0, sticky="w", padx=12)

        ttk.Checkbutton(
            frame,
            text="Normalize seismic amplitudes to uint16 for TIFF viewing",
            variable=self.normalize,
        ).grid(row=11, column=0, sticky="w", padx=12, pady=8)

        convert_button = ttk.Button(
            frame,
            text="Convert",
            command=self.convert,
        )
        convert_button.grid(row=12, column=0, sticky="w", padx=12, pady=18)

        self.status_label = ttk.Label(frame, text="", foreground="blue")
        self.status_label.grid(row=13, column=0, sticky="w", padx=12)

    def select_input_file(self):
        file_path = filedialog.askopenfilename(
            title="Select seismic .npy file",
            filetypes=[
                ("NumPy files", "*.npy"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            self.input_file.set(file_path)

            input_path = Path(file_path)
            suggested_output = input_path.with_suffix(".tif")
            self.output_path.set(str(suggested_output))

    def select_output_path(self):
        if self.export_mode.get() == "stack":
            file_path = filedialog.asksaveasfilename(
                title="Save TIFF file",
                defaultextension=".tif",
                filetypes=[
                    ("TIFF files", "*.tif *.tiff"),
                    ("All files", "*.*"),
                ],
            )

            if file_path:
                self.output_path.set(file_path)

        else:
            folder_path = filedialog.askdirectory(
                title="Select folder for TIFF slices"
            )

            if folder_path:
                self.output_path.set(folder_path)

    def update_output_label(self):
        self.output_path.set("")

    def convert(self):
        input_file = self.input_file.get().strip()
        output_path = self.output_path.get().strip()

        if not input_file:
            messagebox.showerror("Missing input", "Please select an input .npy file.")
            return

        if not output_path:
            messagebox.showerror("Missing output", "Please select an output file or folder.")
            return

        input_path = Path(input_file)

        if not input_path.exists():
            messagebox.showerror("File not found", "The selected input file does not exist.")
            return

        if input_path.suffix.lower() != ".npy":
            messagebox.showerror("Invalid file", "Please select a .npy file.")
            return

        try:
            self.status_label.config(text="Loading .npy file...")
            self.root.update_idletasks()

            arr = np.load(input_path)

            if arr.ndim not in [2, 3]:
                raise ValueError(
                    f"Only 2D or 3D arrays are supported. Found array shape: {arr.shape}"
                )

            self.status_label.config(
                text=f"Loaded shape {arr.shape}, dtype {arr.dtype}. Converting..."
            )
            self.root.update_idletasks()

            normalize = self.normalize.get()
            rotation_degrees = self.rotation.get()

            if self.export_mode.get() == "stack":
                out_path = Path(output_path)

                if out_path.suffix.lower() not in [".tif", ".tiff"]:
                    out_path = out_path.with_suffix(".tif")

                save_volume_tiff(
                    arr,
                    out_path,
                    normalize=normalize,
                    rotation_degrees=rotation_degrees,
                )

                messagebox.showinfo(
                    "Success",
                    f"TIFF saved successfully:\n{out_path}",
                )

            else:
                save_slices(
                    arr,
                    output_path,
                    axis=self.axis.get(),
                    normalize=normalize,
                    rotation_degrees=rotation_degrees,
                )

                messagebox.showinfo(
                    "Success",
                    f"TIFF slices saved successfully:\n{output_path}",
                )

            self.status_label.config(text="Conversion complete.")

        except Exception as e:
            self.status_label.config(text="Conversion failed.")
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    app = NpyToTiffGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()