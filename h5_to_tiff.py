import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import h5py
import numpy as np
from PIL import Image
from skimage import exposure as exp


def normalize_to_uint8(image, low_percentile=1.0, high_percentile=99.0):
    image = np.asarray(image)

    pmin, pmax = np.percentile(image, (low_percentile, high_percentile))

    if pmax == pmin:
        return np.zeros(image.shape, dtype=np.uint8)

    image = exp.rescale_intensity(
        image,
        in_range=(pmin, pmax),
        out_range=(0, 255)
    )

    return image.astype(np.uint8)


def label_to_uint8(label_image):
    label_image = np.asarray(label_image)

    if label_image.max() <= 7:
        label_image = label_image * 32

    return np.clip(label_image, 0, 255).astype(np.uint8)


def convert_h5_to_tiff_stack(
    h5_path,
    output_tiff_path,
    dataset_name,
    start_index,
    end_index,
    channel_index,
    save_mode,
    progress_callback,
    log_callback
):
    frames = []

    with h5py.File(h5_path, "r") as f:
        if dataset_name not in f:
            raise KeyError(
                f"Dataset '{dataset_name}' not found. "
                f"Available keys: {list(f.keys())}"
            )

        data = f[dataset_name]
        total_slices = data.shape[0]

        if end_index is None or end_index >= total_slices:
            end_index = total_slices - 1

        if start_index < 0 or start_index >= total_slices:
            raise ValueError(f"Start index must be between 0 and {total_slices - 1}")

        if end_index < start_index:
            raise ValueError("End index must be greater than or equal to start index.")

        count = end_index - start_index + 1

        log_callback(f"HDF5 file: {h5_path}")
        log_callback(f"Dataset: {dataset_name}")
        log_callback(f"Shape: {data.shape}")
        log_callback(f"Saving TIFF stack: {output_tiff_path}")
        log_callback(f"Slices: {start_index} to {end_index}")
        log_callback("")

        for n, i in enumerate(range(start_index, end_index + 1), start=1):
            if dataset_name == "features":
                if data.ndim == 4:
                    image = data[i, :, :, channel_index]
                elif data.ndim == 3:
                    image = data[i, :, :]
                else:
                    raise ValueError(f"Unsupported features shape: {data.shape}")

                if save_mode == "uint8_normalized":
                    image = normalize_to_uint8(image)
                elif save_mode == "float32":
                    image = np.asarray(image, dtype=np.float32)
                else:
                    raise ValueError(f"Unknown save mode: {save_mode}")

            elif dataset_name == "label":
                image = data[i, :, :]
                image = label_to_uint8(image)

            else:
                if data.ndim == 4:
                    image = data[i, :, :, channel_index]
                elif data.ndim == 3:
                    image = data[i, :, :]
                else:
                    raise ValueError(f"Unsupported dataset shape: {data.shape}")

                if np.issubdtype(image.dtype, np.floating):
                    image = normalize_to_uint8(image)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)

            frames.append(Image.fromarray(image))

            progress = int((n / count) * 90)
            progress_callback(progress)
            log_callback(f"Loaded slice {i}")

        if not frames:
            raise ValueError("No slices were loaded.")

        log_callback("")
        log_callback("Writing TIFF stack...")

        frames[0].save(
            output_tiff_path,
            save_all=True,
            append_images=frames[1:],
            compression="tiff_deflate"
        )

        progress_callback(100)
        log_callback("TIFF stack saved successfully.")


class H5ToTiffStackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HDF5 Seismic Data to TIFF Stack Converter")
        self.root.geometry("760x540")

        self.h5_path = tk.StringVar()
        self.output_tiff_path = tk.StringVar()
        self.dataset_name = tk.StringVar(value="features")
        self.start_index = tk.StringVar(value="0")
        self.end_index = tk.StringVar(value="")
        self.channel_index = tk.StringVar(value="0")
        self.save_mode = tk.StringVar(value="uint8_normalized")

        self.build_gui()

    def build_gui(self):
        padding = {"padx": 10, "pady": 6}

        frame = ttk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Input .h5 file").grid(row=0, column=0, sticky="w", **padding)
        ttk.Entry(frame, textvariable=self.h5_path, width=72).grid(row=0, column=1, **padding)
        ttk.Button(frame, text="Browse", command=self.choose_h5_file).grid(row=0, column=2, **padding)

        ttk.Label(frame, text="Output TIFF stack").grid(row=1, column=0, sticky="w", **padding)
        ttk.Entry(frame, textvariable=self.output_tiff_path, width=72).grid(row=1, column=1, **padding)
        ttk.Button(frame, text="Save as", command=self.choose_output_tiff).grid(row=1, column=2, **padding)

        ttk.Label(frame, text="Dataset").grid(row=2, column=0, sticky="w", **padding)
        ttk.Combobox(
            frame,
            textvariable=self.dataset_name,
            values=["features", "label"],
            state="readonly",
            width=22
        ).grid(row=2, column=1, sticky="w", **padding)

        ttk.Label(frame, text="Start slice index").grid(row=3, column=0, sticky="w", **padding)
        ttk.Entry(frame, textvariable=self.start_index, width=22).grid(row=3, column=1, sticky="w", **padding)

        ttk.Label(frame, text="End slice index").grid(row=4, column=0, sticky="w", **padding)
        ttk.Entry(frame, textvariable=self.end_index, width=22).grid(row=4, column=1, sticky="w", **padding)
        ttk.Label(frame, text="Leave blank for last slice").grid(row=4, column=1, sticky="e", **padding)

        ttk.Label(frame, text="Channel index").grid(row=5, column=0, sticky="w", **padding)
        ttk.Entry(frame, textvariable=self.channel_index, width=22).grid(row=5, column=1, sticky="w", **padding)
        ttk.Label(frame, text="Use 0 for Penobscot features").grid(row=5, column=1, sticky="e", **padding)

        ttk.Label(frame, text="Save mode").grid(row=6, column=0, sticky="w", **padding)
        ttk.Combobox(
            frame,
            textvariable=self.save_mode,
            values=["uint8_normalized", "float32"],
            state="readonly",
            width=22
        ).grid(row=6, column=1, sticky="w", **padding)

        ttk.Button(
            frame,
            text="Convert to TIFF Stack",
            command=self.start_conversion
        ).grid(row=7, column=1, sticky="w", **padding)

        self.progress = ttk.Progressbar(frame, orient="horizontal", length=580, mode="determinate")
        self.progress.grid(row=8, column=0, columnspan=3, sticky="w", **padding)

        ttk.Label(frame, text="Log").grid(row=9, column=0, sticky="w", **padding)

        self.log_text = tk.Text(frame, height=14, width=90)
        self.log_text.grid(row=10, column=0, columnspan=3, padx=10, pady=6)

    def choose_h5_file(self):
        path = filedialog.askopenfilename(
            title="Choose HDF5 file",
            filetypes=[
                ("HDF5 files", "*.h5 *.hdf5"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.h5_path.set(path)

            base = os.path.splitext(os.path.basename(path))[0]
            suggested = os.path.join(os.path.dirname(path), f"{base}_stack.tiff")
            self.output_tiff_path.set(suggested)

    def choose_output_tiff(self):
        path = filedialog.asksaveasfilename(
            title="Save TIFF stack as",
            defaultextension=".tiff",
            filetypes=[
                ("TIFF stack", "*.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.output_tiff_path.set(path)

    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.root.update_idletasks()

    def update_progress(self, value):
        self.progress["value"] = value
        self.root.update_idletasks()

    def start_conversion(self):
        h5_path = self.h5_path.get().strip()
        output_tiff_path = self.output_tiff_path.get().strip()
        dataset_name = self.dataset_name.get().strip()

        if not h5_path:
            messagebox.showerror("Missing file", "Please choose an input .h5 file.")
            return

        if not os.path.exists(h5_path):
            messagebox.showerror("File not found", f"File does not exist:\n{h5_path}")
            return

        if not output_tiff_path:
            messagebox.showerror("Missing output file", "Please choose an output TIFF stack file.")
            return

        try:
            start_index = int(self.start_index.get())
        except ValueError:
            messagebox.showerror("Invalid start index", "Start index must be an integer.")
            return

        end_text = self.end_index.get().strip()
        if end_text == "":
            end_index = None
        else:
            try:
                end_index = int(end_text)
            except ValueError:
                messagebox.showerror("Invalid end index", "End index must be an integer or blank.")
                return

        try:
            channel_index = int(self.channel_index.get())
        except ValueError:
            messagebox.showerror("Invalid channel index", "Channel index must be an integer.")
            return

        save_mode = self.save_mode.get()

        self.progress["value"] = 0
        self.log_text.delete("1.0", "end")

        thread = threading.Thread(
            target=self.run_conversion,
            args=(
                h5_path,
                output_tiff_path,
                dataset_name,
                start_index,
                end_index,
                channel_index,
                save_mode
            ),
            daemon=True
        )
        thread.start()

    def run_conversion(
        self,
        h5_path,
        output_tiff_path,
        dataset_name,
        start_index,
        end_index,
        channel_index,
        save_mode
    ):
        try:
            convert_h5_to_tiff_stack(
                h5_path=h5_path,
                output_tiff_path=output_tiff_path,
                dataset_name=dataset_name,
                start_index=start_index,
                end_index=end_index,
                channel_index=channel_index,
                save_mode=save_mode,
                progress_callback=self.update_progress,
                log_callback=self.log
            )

            messagebox.showinfo("Done", "TIFF stack conversion completed successfully.")

        except Exception as e:
            self.log(f"ERROR: {e}")
            messagebox.showerror("Conversion error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = H5ToTiffStackGUI(root)
    root.mainloop()