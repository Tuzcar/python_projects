#-------------------------------------------------------------------------------
# Name:        PaniniP
# Purpose:     Image Processing code to achieve Panini perspective in images
#
# Author:      Montserrat Ordaz
#
# Created:     05/05/2025
# Copyright:   (c) monts 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import cv2
import time
##import numpy as np
from tkinter import Tk, Button, Label
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

class ImageApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
##        self.cap = cv2.VideoCapture(0)
        self.cap = cv2.imread(r"C:\Users\monts\Pictures\bocadeIguanas0.png",0)
        self.playing = False



        # Display frame
        self.label = Label(window)
        self.label.pack()

        # Buttons
        Button(window, text="Play/Stop", width=10, command=self.toggle_play).pack(side="left")
##        Button(window, text="Capture", width=10, command=self.upload).pack(side="left")
        Button(window, text="Capture", width=10, command=self.capture).pack(side="left")
        Button(window, text="Show All", width=10, command=self.show_all).pack(side="left")
        Button(window, text="Picture", width=10, command=self.show_original).pack(side="left")
        Button(window, text="B&W", width=10, command=self.show_bw).pack(side="left")
        Button(window, text="Laplacian", width=10, command=self.show_lap).pack(side="left")
        Button(window, text="Sobel X", width=10, command=self.show_sx).pack(side="left")
        Button(window, text="Sobel Y", width=10, command=self.show_sy).pack(side="left")

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.mainloop()

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.update_frame()

    def update_frame(self):
        if self.playing:
##            ret, frame = self.upl()
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img_pil)
                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)
            self.window.after(30, self.update_frame)

##    def upload(self):
##        ret, frame = self.upl.cv2.imread(r"C:\Users\monts\Pictures\bocadeIguanas0.png")
##        if ret:
##            cv2.imwrite('IMGtoEdit.png', frame)
##            print("Saved to IMGtoEdit.png")


    def capture(self):
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite('IMGtoEdit.png', frame)
            print("Captured to IMGtoEdit.png")

    def show_original(self):
        image = cv2.imread('IMGtoEdit.png')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')
        plt.show()
        cv2.waitKey(0)
        self._save_fig('Orig')

    def show_bw(self):
        img = cv2.imread('IMGtoEdit.png', 0)
        plt.imshow(img, cmap='gray')
        plt.title('B&W')
        plt.axis('off')
        plt.show()
        self._save_fig('BW')

    def show_lap(self):
        img = cv2.imread('IMGtoEdit.png', 0)
        lap = cv2.Laplacian(img, cv2.CV_64F)
        plt.imshow(lap, cmap='gray')
        plt.title('Laplacian')
        plt.axis('off')
        plt.show()
        self._save_fig('Lap')

    def show_sx(self):
        img = cv2.imread('IMGtoEdit.png', 0)
        sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        plt.imshow(sx, cmap='gray')
        plt.title('Sobel X')
        plt.axis('off')
        plt.show()
        self._save_fig('SX')

    def show_sy(self):
        img = cv2.imread('IMGtoEdit.png', 0)
        sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        plt.imshow(sy, cmap='gray')
        plt.title('Sobel Y')
        plt.axis('off')
        plt.show()
        self._save_fig('SY')

    def show_all(self):
        image = cv2.imread('IMGtoEdit.png')
        img_gray = cv2.imread('IMGtoEdit.png', 0)
        lap = cv2.Laplacian(img_gray, cv2.CV_64F)
        sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
        sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)

        plt.figure(figsize=(8,6))
        plt.subplot(2,3,1); plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.title('Original'); plt.axis('off')
        plt.subplot(2,3,4); plt.imshow(img_gray, cmap='gray'); plt.title('BW'); plt.axis('off')
        plt.subplot(2,3,2); plt.imshow(lap, cmap='gray'); plt.title('Laplacian'); plt.axis('off')
        plt.subplot(2,3,3); plt.imshow(sx, cmap='gray'); plt.title('Sobel X'); plt.axis('off')
        plt.subplot(2,3,6); plt.imshow(sy, cmap='gray'); plt.title('Sobel Y'); plt.axis('off')
        plt.show()

    def _save_fig(self, prefix):
        timestr = time.strftime("%d%m%Y_%H%M%S")
        fig = plt.gcf()
        fig.savefig(f'{prefix}_{timestr}.png', dpi=100)
        print(f"Saved {prefix}_{timestr}.png")

    def on_close(self):
        self.playing = False
##        self.upl.release()
        self.cap.release()
        self.window.destroy()

if __name__ == '__main__':
    ImageApp(Tk(), "Image Processing App")
