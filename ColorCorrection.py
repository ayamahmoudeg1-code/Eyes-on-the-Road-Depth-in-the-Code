import cv2 as cv
import numpy as np
import sys
#Reading the input image as a grayscale
org_image = cv.imread('side quest unedited image.png')

#conerting to lab to hav emore control over color channels
lab = cv.cvtColor(org_image, cv.COLOR_BGR2Lab)
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
    cv.imwrite('Edited image.png', enhanced_img )


cv.waitKey(0) # waiting forever till the user exit 
cv.destroyAllWindows() # destroying all windows upon exiting 
