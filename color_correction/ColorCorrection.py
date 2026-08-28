import cv2 as cv
import numpy as np
import sys
#Reading the input image as a grayscale
org_image = cv.imread('color_correction\side quest unedited image.png')

b, g, r = cv.split(org_image)

#calculate average channel intinsity 
mean_b = np.mean(b)
mean_g = np.mean(g)
mean_r = np.mean(r)

mean_gray = (mean_b + mean_g + mean_r) / 3.0
#apllying White Balance to all channels
b_wb = np.clip(b * (mean_gray / mean_b), 0, 255).astype(np.uint8)
g_wb = np.clip(g * (mean_gray / mean_g), 0, 255).astype(np.uint8)
r_wb = np.clip(r * (mean_gray / mean_r), 0, 255).astype(np.uint8)


# merge channels back toghether
wb_image = cv.merge([b_wb, g_wb, r_wb])

#conerting to lab to hav emore control over color channels
lab = cv.cvtColor(wb_image, cv.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv.split(lab)

#applying clahe which enhnces contrast
clahe = cv.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)
#applying it only to the white and black channel(l channel)
enhanced = clahe.apply(l_channel)

#merging new changes
merged_lab = cv.merge((enhanced, a_channel, b_channel))

# reconverting it back to original
enhanced_img = cv.cvtColor(merged_lab, cv.COLOR_Lab2BGR)

cv.imshow('Original image', org_image)  # displaying original image
cv.imshow('Edited image', enhanced_img )  # display edited image
k = cv.waitKey(0)  # wait for a key press

# saving the output image by presssing 's' key
if k == ord('s'):
    cv.imwrite('Final Edited image.png', enhanced_img )


cv.waitKey(0) # waiting forever till the user exit 
cv.destroyAllWindows() # destroying all windows upon exiting 
