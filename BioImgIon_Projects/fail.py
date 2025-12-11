#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      monts
#
# Created:     09/12/2025
# Copyright:   (c) monts 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import signal
from mpl_toolkits.axes_grid1 import ImageGrid

np.seterr(divide='ignore', invalid='ignore')

#To load matlab files v7.3 HDF5

def load_h5(path, varname):
    with h5py.File(path, "r") as f:
        arr = np.array(f[varname])
    return np.squeeze(arr)


#Task 1 – Contrast to Noise Ratio (CNR)

#T1a) ***********************
data = load_h5('Ion_Assignment_4_CNRData_1D.mat',"data_noise")

plt.figure(figsize=(12,3))
plt.plot(data)
plt.title("1D noisy data")
plt.xlabel("Pixel number")
plt.ylabel("Intensity")
plt.grid(True, linestyle=":")
plt.show()

#T1b)*********************
objects = {
    "A": (100, 300),
    "B": (400, 600),
    "C": (700, 900),
}
#background
bg = np.concatenate([data[:100], data[-100:]])#first 100 and last 100 pixels
bg_mean = np.mean(bg)
bg_std  = np.std(bg)

print("/n")
print("Background mean:", bg_mean)
print("Background std :", bg_std)

#computing CNRs
for obj, (s,e) in objects.items():
    obj_mean = np.mean(data[s:e])
    cnr = (obj_mean - bg_mean) / bg_std
    print(f"Object {obj}: /n mean={obj_mean:.3f}, /n CNR={cnr:.3f}")



#Task 2 – Modulation Transfer Function (MTF)
#T2a)**********************************
pattern = load_h5("Ion_Assignment_4_MTFData_Sin_2D.mat", "pattern")

plt.figure(figsize=(10,4))
plt.imshow(pattern, cmap="gray", aspect="auto")
plt.title("Reference test pattern")
plt.xlabel("x (microm)")
plt.ylabel("y")
plt.colorbar()
plt.show()

#to extract central row
center_row = pattern[pattern.shape[0]//2, :]

plt.figure(figsize=(12,3))
plt.plot(center_row)
plt.title("Central row of test pattern")
plt.xlabel("x (microm)")
plt.ylabel("Intensity")
plt.grid(True, which="both", linestyle=":")
plt.show()

#T2b)**********************
#Convolution with gaussian PSF
psf = signal.windows.gaussian(pattern.shape[1], std=50)
psf = psf / np.sum(psf)  # normalize

#Convolve along x-direction
imaged = signal.fftconvolve(
    pattern,
    np.expand_dims(psf, axis=0),
    axes=1,
    mode="same"
)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.imshow(pattern, cmap="gray", aspect="auto")
plt.title("Original pattern")

plt.subplot(1,2,2)
plt.imshow(imaged, cmap="gray", aspect="auto")
plt.title("After convolution with PSF")
plt.show()

# Overlay central rows
conv_row = imaged[pattern.shape[0]//2,:]

plt.figure(figsize=(12,3))
plt.plot(center_row, label="Original")
plt.plot(conv_row, label="Convolved")
plt.title("Central row before/after imaging system")
plt.xlabel("x (micrometer)")
plt.ylabel("Intensity")
plt.grid(True)
plt.legend()
plt.show()



#Task 3 – Digital Subtraction Angiography (DSA)
#a)****************
dsa = load_h5("Ion_Assignment_4_DSAData.mat","dsa_series")

#HDF5 reads the matlab files reversed (matlab stores images as (x,y,frames))
#swap to (y,x,frames) for correct orientation
dsa = np.transpose(dsa, (1,0,2))

ny, nx, nf = dsa.shape
print("\n")
print("DSA shape:",dsa.shape)

fig = plt.figure(figsize=(12, 2.5*(nf//6 + 1)))
grid = ImageGrid(fig, 111, nrows_ncols=(int(np.ceil(nf/6)), 6), axes_pad=0.0)

for i, ax in enumerate(grid):
    if i < nf:
        ax.imshow(dsa[:,:,i],cmap="gray")
        ax.set_title(f"Frame {i}")
    ax.axis("off")

plt.show()

#T2b)
#DSA images: ln(I0) - ln(Ii)
I0 = dsa[:,:,0]
I0[I0<=0] = 1e-6

dsa_imgs = np.zeros_like(dsa)

for i in range(nf):
    Ii = dsa[:,:,i]
    Ii[Ii<=0] = 1e-6
    dsa_imgs[:,:,i] = np.log(I0) - np.log(Ii)



##
##
##import numpy as np
##import matplotlib.pyplot as plt
##import scipy.io as sio
##import h5py
##from scipy import signal
##from mpl_toolkits.axes_grid1 import ImageGrid
##
##def load_mat_v73(path):
##    with h5py.File(path, 'r') as f:
##        return {k: f[k][()] for k in f.keys()}
##
##np.seterr(divide='ignore', invalid='ignore')
##
### =====================================================
### Task 1 – Contrast to Noise Ratio (CNR)
### =====================================================
##
### (a) Load and plot 1D data
####mat_cnr = h5py.File('Ion_Assignment_4_CNRData_1D.mat')
####data = np.array(f['data_noise'])
####data = np.squeeze(data)
##
##with h5py.File('Ion_Assignment_4_CNRData_1D.mat', 'r') as f:
##    data = np.array(f['data_noise'])
##    data = np.squeeze(data)
##
##plt.figure(figsize=(12,3))
##plt.plot(data)
##plt.title("1D noisy data")
##plt.xlabel("Pixel number")
##plt.ylabel("Intensity")
##plt.grid(True, linestyle=":")
##plt.show()
##
### (b) Extract object & background data
##objects = {
##    "A": (100, 300),
##    "B": (400, 600),
##    "C": (700, 900),
##}
##
### background: first 100 and last 100 pixels
##bg = np.concatenate([data[:100], data[-100:]])
##bg_mean = np.mean(bg)
##bg_std  = np.std(bg)
##
##print("\nBackground mean:", bg_mean)
##print("Background std :", bg_std)
##
### Compute CNRs
##for obj, (s,e) in objects.items():
##    obj_mean = np.mean(data[s:e])
##    cnr = (obj_mean - bg_mean) / bg_std
##    print(f"Object {obj}: mean={obj_mean:.3f}, CNR={cnr:.3f}")
##
### =====================================================
### Task 2 – Modulation Transfer Function (MTF)
### =====================================================
##
### (a) Load pattern and extract central row
##mat_mtf = h5py.File("Ion_Assignment_4_MTFData_Sin_2D.mat")
##pattern = np.squeeze(mat_mtf["pattern"])
##
##plt.figure(figsize=(10,4))
##plt.imshow(pattern, cmap="gray", aspect="auto")
##plt.title("Reference test pattern")
##plt.xlabel("x (microm)")
##plt.ylabel("y")
##plt.colorbar()
##plt.show()
##
### central row
##center_row = pattern[pattern.shape[0]//2, :]
##
##plt.figure(figsize=(12,3))
##plt.plot(center_row)
##plt.title("Central row of test pattern")
##plt.xlabel("x (microm)")
##plt.ylabel("Intensity")
##plt.grid(True, which="both", linestyle=":")
##plt.show()
##
### COMMENT:
### Visually inspect the number of cycles inside each block.
### The 4 blocks have increasing spatial frequency.
### Write cy/mm values here as a comment in your code.
##
### (b) Convolve with Gaussian PSF and display
##psf = signal.windows.gaussian(pattern.shape[1], std=50)
##psf = psf / np.sum(psf)  # normalize
##
### Convolve only along x-direction
##imaged = signal.fftconvolve(
##    pattern,
##    np.expand_dims(psf, axis=0),
##    axes=1,
##    mode="same"
##)
##
##plt.figure(figsize=(12,4))
##plt.subplot(1,2,1)
##plt.imshow(pattern, cmap="gray", aspect="auto")
##plt.title("Original pattern")
##
##plt.subplot(1,2,2)
##plt.imshow(imaged, cmap="gray", aspect="auto")
##plt.title("After convolution with PSF")
##plt.show()
##
### Overlay central rows
##conv_row = imaged[pattern.shape[0]//2, :]
##
##plt.figure(figsize=(12,3))
##plt.plot(center_row, label="Original")
##plt.plot(conv_row, label="Convolved")
##plt.title("Central row before/after imaging system")
##plt.xlabel("x (microm)")
##plt.ylabel("Intensity")
##plt.grid(True)
##plt.legend()
##plt.show()
##
### =====================================================
### Task 3 – Digital Subtraction Angiography (DSA)
### =====================================================
##
### (a) Load & display all fluoroscopy images
##mat_dsa = h5py.File("Ion_Assignment_4_DSAData.mat")
##dsa = np.array(mat_dsa["dsa_series"], dtype="float32")
##
##ny, nx, nf = dsa.shape
##print("\nDSA shape:", dsa.shape)
##
##fig = plt.figure(figsize=(12, 2.5*(nf//6 + 1)))
##grid = ImageGrid(fig, 111, nrows_ncols=(int(np.ceil(nf/6)), 6), axes_pad=0.0)
##
##for i, ax in enumerate(grid):
##    if i < nf:
##        ax.imshow(dsa[:,:,i], cmap="gray")
##        ax.set_title(f"Frame {i}")
##    ax.axis("off")
##
##plt.show()
##
### (b) Compute DSA images: ln(I0) - ln(Ii)
##I0 = dsa[:,:,0]
##I0[I0<=0] = 1e-6
##
##dsa_imgs = np.zeros_like(dsa)
##
##for i in range(nf):
##    Ii = dsa[:,:,i]
##    Ii[Ii<=0] = 1e-6
##    dsa_imgs[:,:,i] = np.log(I0) - np.log(Ii)
