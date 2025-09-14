import os
import sys
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk
import cv2


def _here(*paths):
    return os.path.join(os.path.dirname(__file__), *paths)


class LauncherUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HTN25 Launcher")
        # Pin window on top
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass

        # Fixed small size
        try:
            self.root.resizable(False, False)
            # Very small window at top-left corner
            self.root.geometry("120x90+0+0")
        except Exception:
            pass

        self.calib_proc = None
        self.demo_proc = None
        self.grid_thread = None
        self.grid_stop = None

        frame = ttk.Frame(self.root, padding=6)
        frame.pack(fill=tk.BOTH, expand=True)

        # Two small square buttons side-by-side
        frame.columnconfigure(0, weight=1, uniform="b")
        frame.columnconfigure(1, weight=1, uniform="b")
        frame.rowconfigure(0, weight=1)

        # Use tk.Button to control width/height in text units (approx square)
        self.btn_calib = tk.Button(frame, text="Cal", width=4, height=2, command=self.start_calibration)
        self.btn_calib.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")

        self.btn_demo = tk.Button(frame, text="Demo", width=4, height=2, command=self.start_demo)
        self.btn_demo.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")

        # Handle window close
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        except Exception:
            pass

        # Periodic poll to update button states/text
        self.root.after(300, self._poll_children)

    def _poll_children(self):
        # Update states based on running child processes
        calib_alive = (self.calib_proc is not None) and (self.calib_proc.poll() is None)
        demo_alive = (self.demo_proc is not None) and (self.demo_proc.poll() is None)

        # Keep buttons enabled for toggle; update labels to reflect stop state
        self.btn_calib.configure(text=("Stop Cal" if calib_alive else "Cal"), state=tk.NORMAL)
        self.btn_demo.configure(text=("Stop Demo" if demo_alive else "Demo"), state=tk.NORMAL)

        # If grid thread requested stop (e.g., 'q' pressed), stop calibration process
        if self.grid_stop is not None and self.grid_stop.is_set() and calib_alive:
            self._stop_process("calib_proc")

        # If calibration finished, close grid window and auto-start demo (if not already running)
        if (not calib_alive) and (self.calib_proc is not None):
            if self.grid_stop is not None:
                try:
                    self.grid_stop.set()
                except Exception:
                    pass
                # Let the grid thread exit; we don't need to join here
            # Autostart demo if it's not running
            if not demo_alive:
                self._start_demo_subprocess()
            # Clear reference so we don't re-trigger
            self.calib_proc = None

        self.root.after(500, self._poll_children)

    def _stop_process(self, proc_attr, wait_timeout=0.8):
        proc = getattr(self, proc_attr, None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=wait_timeout)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            setattr(self, proc_attr, None)

    def _generate_or_load_grid(self):
        here = os.path.dirname(__file__)
        try:
            try:
                from create_aruco_png import generate_aruco_corners_image  # when run from within new_rasp/
            except Exception:
                from new_rasp.create_aruco_png import generate_aruco_corners_image  # when run from repo root
            return generate_aruco_corners_image()
        except Exception:
            fallback = os.path.join(here, "aruco_grid.png")
            if not os.path.exists(fallback):
                raise
            return fallback

    def _show_fullscreen_grid(self, image_path, stop_event):
        img = cv2.imread(image_path)
        if img is None:
            return
        win = "Projector Grid"
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
        try:
            while not stop_event.is_set():
                cv2.imshow(win, img)
                key = cv2.waitKey(16) & 0xFF
                if key == ord('q'):
                    stop_event.set()
                    break
        finally:
            try:
                cv2.destroyWindow(win)
            except Exception:
                pass

    def start_calibration(self):
        # Toggle: if calibration is running, stop it
        if (self.calib_proc is not None) and (self.calib_proc.poll() is None):
            self._stop_process("calib_proc")
            # Stop fullscreen grid if showing
            if self.grid_stop is not None:
                try:
                    self.grid_stop.set()
                except Exception:
                    pass
            return
        # Ensure only one instance total: stop demo if running
        if (self.demo_proc is not None) and (self.demo_proc.poll() is None):
            self._stop_process("demo_proc")
        # Start fullscreen grid
        try:
            grid_path = self._generate_or_load_grid()
            self.grid_stop = threading.Event()
            self.grid_thread = threading.Thread(target=self._show_fullscreen_grid, args=(grid_path, self.grid_stop), daemon=True)
            self.grid_thread.start()
        except Exception:
            self.grid_stop = None
            self.grid_thread = None
        script = _here("calibrate_projector_corners_stereo.py")
        # Launch using Windows 'py' per user preference
        try:
            self.calib_proc = subprocess.Popen(["py", script], cwd=os.path.dirname(script))
        except Exception:
            pass

    def start_demo(self):
        # Toggle: if demo is running, stop it
        if (self.demo_proc is not None) and (self.demo_proc.poll() is None):
            self._stop_process("demo_proc")
            return
        # Ensure only one instance total: stop calibration if running
        if (self.calib_proc is not None) and (self.calib_proc.poll() is None):
            self._stop_process("calib_proc")
        # Ensure grid is closed if open
        if self.grid_stop is not None:
            try:
                self.grid_stop.set()
            except Exception:
                pass
        script = _here("stereo_projection_demo.py")
        try:
            self.demo_proc = subprocess.Popen(["py", script], cwd=os.path.dirname(script))
        except Exception:
            pass

    def _start_demo_subprocess(self):
        # Helper that starts demo without toggling/stop logic, used after calibration
        if (self.demo_proc is not None) and (self.demo_proc.poll() is None):
            return
        # Ensure grid is closed
        if self.grid_stop is not None:
            try:
                self.grid_stop.set()
            except Exception:
                pass
        script = _here("stereo_projection_demo.py")
        try:
            self.demo_proc = subprocess.Popen(["py", script], cwd=os.path.dirname(script))
        except Exception:
            pass

    def on_close(self):
        # Stop any running children before closing
        self._stop_process("calib_proc")
        self._stop_process("demo_proc")
        if self.grid_stop is not None:
            try:
                self.grid_stop.set()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass


def main():
    root = tk.Tk()
    LauncherUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())


