import numpy as np
import cv2

def gkern(l=5, sig=1.0):
    ax = np.arange(-l // 2 + 1.0, l // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    out_kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sig ** 2))
    out_kernel = out_kernel / np.sum(out_kernel)
    return out_kernel

def denoise_gauss(image):
    kernel = gkern(5, 1.0)
    denoised_image = np.zeros_like(image)
    for c in range(3):
        denoised_image[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
    return denoised_image
