import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv.imread('dashcam.jpg')
h, w, c = img.shape

# SEEDS requires HSV (or Lab) for iterate
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Create & run SEEDS
seeds = cv.ximgproc.createSuperpixelSEEDS(
    w, h, c,
    num_superpixels=200,
    num_levels=4,
    prior=2,
    histogram_bins=5
)
seeds.iterate(img_hsv, 6)

# Get contour mask (1 = boundary)
contour = seeds.getLabelContourMask(thickLine=True)

# Overlay contours on original
result = img.copy()
result[contour == 1] = [0, 0, 0]  # black boundaries

# Convert BGR → RGB for matplotlib
img_rgb  = cv.cvtColor(img,    cv.COLOR_BGR2RGB)
res_rgb  = cv.cvtColor(result, cv.COLOR_BGR2RGB)

# Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img_rgb)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(res_rgb)
axes[1].set_title("SEEDS Superpixels")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("seeds_comparison.png", dpi=150)
plt.show()   