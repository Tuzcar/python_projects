#-------------------------------------------------------------------------------
# Name:        PaniniProjection_module
# Purpose:
#
# Author:      Montserrat Ordaz
#
# Created:     12/05/2025
# Copyright:   (c) monts 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import cv2
import time
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

class PaniniProjectionBase:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Capture available camera
        self.cap = cv2.VideoCapture(0)
        self.playing = False

        #to upload existent image
        self.img_path = None

        # Display frame
        self.label = Label(window)
        self.label.pack()

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.update_frame()

    def _update_frame(self):
        if not self.playing:
            return

        ret, frame = self.cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.window.after(30, self.update_frame)

    def upload(self):
        """Open files and set an image picked by user"""
        pass

    def capture(self):
        pass

    def show_original(self):
        pass

    def show_bw(self):
        pass

    def show_lap(self):
        pass

    def show_all(self):
        pass

    def _save_fig(self, prefix):
        timestr = time.strftime("%d%m%Y_%H%M%S")
        fig = plt.gcf()
        fname = f'{prefix}_{timestr}.png'
        fig.savefig(fname, dpi=100)
        print(f"Saved {fname}")

    def on_close(self):
        self.playing = False
        self.cap.release()
        self.window.destroy()

if __name__ == '__main__':
    PaniniProjectionBase(Tk(), "Image Processing App")