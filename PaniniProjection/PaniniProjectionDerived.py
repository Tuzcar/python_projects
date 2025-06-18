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
from numpy import asarray
from PaniniProjectionBase import PaniniProjectionBase

class ImageApp(PaniniProjectionBase):
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

        # Variable that will store the image as array
        self.arrayImage = []

        # Buttons
        Button(window, text="Play/Stop",  width=10, command=self._toggle_play).pack(side="left")
        Button(window, text="Capture",    width=10, command=self.capture).pack(side="left")
        Button(window, text="Upload",     width=10, command=self.upload).pack(side="left")
        Button(window, text="Show All",   width=10, command=self.show_all).pack(side="left")
        Button(window, text="Original",   width=10, command=self.show_original).pack(side="left")
        Button(window, text="B&W",        width=10, command=self.show_bw).pack(side="left")
        Button(window, text="Laplacian",  width=10, command=self.show_lap).pack(side="left")
##        Button(window, text="Sobel X",    width=10, command=self.show_sx).pack(side="left")
##        Button(window, text="Sobel Y",    width=10, command=self.show_sy).pack(side="left")

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.window.mainloop()

    def upload(self):
        """Open files and set an image picked by user"""
        path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        self.img_path = path

        # Shows image on the UI
        cv_img = cv2.imread(self.img_path)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.arrayImage = asarray(cv_img)
        print(self.arrayImage.shape)
        img_pil = Image.fromarray(cv_img)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

    def capture(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        cv2.imwrite('IMGtoEdit.png', frame)
        print("Captured to IMGtoEdit.png")
        self.img_path = 'IMGtoEdit.png'

    def _load(self, flags=cv2.IMREAD_COLOR): #takes img from upload or capture
        if not self.img_path:
            raise RuntimeError("No image loaded")
        return cv2.imread(self.img_path, flags)

    def show_original(self):
        image = self._load(cv2.IMREAD_COLOR)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')
        plt.show()
        self._save_fig('Orig')

    def show_bw(self):
        img = self._load(cv2.IMREAD_GRAYSCALE)
        plt.imshow(img, cmap='gray')
        plt.title('B&W')
        plt.axis('off')
        plt.show()
        self._save_fig('BW')

    def show_lap(self):
        img = self._load(cv2.IMREAD_GRAYSCALE)
        lap = cv2.Laplacian(img, cv2.CV_64F)
        plt.imshow(lap, cmap='gray')
        plt.title('Laplacian')
        plt.axis('off')
        plt.show()
        self._save_fig('Lap')

##    def show_sx(self):
##        img = self._load(cv2.IMREAD_GRAYSCALE)
##        sx  = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
##        plt.imshow(sx, cmap='gray')
##        plt.title('Sobel X')
##        plt.axis('off')
##        plt.show()
##        self._save_fig('SX')

##    def show_sy(self):
##        img = self._load(cv2.IMREAD_GRAYSCALE)
##        sy  = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
##        plt.imshow(sy, cmap='gray')
##        plt.title('Sobel Y')
##        plt.axis('off')
##        plt.show()
##        self._save_fig('SY')

    def show_all(self):
        color = self._load(cv2.IMREAD_COLOR)
        gray  = self._load(cv2.IMREAD_GRAYSCALE)
        lap   = cv2.Laplacian(gray, cv2.CV_64F)
##        sx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
##        sy    = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        plt.figure(figsize=(8,6))
        plt.subplot(2,3,1); plt.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)); plt.title('Original'); plt.axis('off')
        plt.subplot(2,3,4); plt.imshow(gray, cmap='gray'); plt.title('B&W'); plt.axis('off')
        plt.subplot(2,3,2); plt.imshow(lap, cmap='gray'); plt.title('Laplacian'); plt.axis('off')
##        plt.subplot(2,3,3); plt.imshow(sx, cmap='gray'); plt.title('Sobel X'); plt.axis('off')
##        plt.subplot(2,3,6); plt.imshow(sy, cmap='gray'); plt.title('Sobel Y'); plt.axis('off')
        plt.show()

if __name__ == '__main__':
    ImageApp(Tk(), "Image Processing App")

