import cv2
import numpy as np

# Load the underwater image
img = cv2.imread("Edited image.png")

# Split into B, G, R channels
b, g, r = cv2.split(img)

# 1. Convert to float32 for precision math
b = b.astype(np.float32)
g = g.astype(np.float32)
r = r.astype(np.float32)

# 2. Compensate for Red channel loss
# Water absorbs Red heavily. We compensate using Green/Blue averages.
mean_b = np.mean(b)
mean_g = np.mean(g)
mean_r = np.mean(r)

# If red is severely attenuated, boost it dynamically relative to green
if mean_r < mean_g:
    r = r + (mean_g - mean_r) * 1.5 

# 3. Histogram Stretching function to normalize each channel individually
def normalize_channel(channel):
    min_val = np.percentile(channel, 1)  # Reject bottom 1% noise
    max_val = np.percentile(channel, 99) # Reject top 1% noise
    stretched = (channel - min_val) * (255.0 / (max_val - min_val + 1e-5))
    return np.clip(stretched, 0, 255)

b_norm = normalize_channel(b)
g_norm = normalize_channel(g)
r_norm = normalize_channel(r)

# Merge back into BGR
balanced_bgr = cv2.merge([b_norm, g_norm, r_norm]).astype(np.uint8)

# 4. Final step: Enhancing contrast via LAB space for sharpness
lab = cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2LAB)
l, la, lb = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_enhanced = clahe.apply(l)

final_img = cv2.cvtColor(cv2.merge([l_enhanced, la, lb]), cv2.COLOR_LAB2BGR)

# Save the real-to-life result
cv2.imwrite("underwater_corrected.jpg", final_img)
