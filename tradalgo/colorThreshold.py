import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the original BGR image
img_bgr = cv2.imread("dashcam.jpg")

# 2. Convert to HSV color space for thresholding
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# 3. Define target color bounds (Red wraps around the hue wheel → two ranges)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# 4. Generate the binary mask (combine both ranges)
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# 5. Isolate target color using the mask
result_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

# 6. Convert OpenCV (BGR) arrays to Matplotlib (RGB) formatting
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

# 7. Construct a single Matplotlib figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Original Asset
axes[0].imshow(img_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Plot 2: Binary Mask (Single-channel requires 'gray' colormap)
axes[1].imshow(mask, cmap="gray")
axes[1].set_title("Binary Mask")
axes[1].axis("off")

# Plot 3: Extracted Threshold Result
axes[2].imshow(result_rgb)
axes[2].set_title("Thresholded Result")
axes[2].axis("off")

# Optimize spatial packing and export the single combined image asset
plt.tight_layout()
plt.savefig("color_threshold_comparison.png", dpi=150)
plt.close()   