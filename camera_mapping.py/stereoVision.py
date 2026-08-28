import cv2 as cv
import numpy as np 

# change the read img as desired 
# CHANGE IN BOTH FILES AND DON'T FORGET TO SAVE !!!
right_img = cv.imread("camera_mapping.py\Frand0.png",0)
left_img = cv.imread("camera_mapping.py\Frand1.png",0)


#creating astereo object with numDisparities which is the max pixel shift between two stereo images
# If an object is so close that its pixel shift exceeds numDisparities, the algorithm will fail to match it and will leave a black hole or artifact in the disparity map.
#blockSize which determine the window size of the images pixeles being compared

nDisparityFactor = 4#changable untill it fits the needs
stereo_object = cv.StereoBM_create(numDisparities= 16*nDisparityFactor, blockSize=15)

#computing disparity map 
disparity_map = stereo_object.compute(right_img, left_img)

# Convert disparity values to visible grayscale
disparity_visual = cv.normalize(
    disparity_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U
)
disparity_colorMap = cv.applyColorMap(disparity_visual, cv.COLORMAP_JET)

cv.imshow("Disparity Map Test", disparity_colorMap)

k = cv.waitKey(0)

if k == ord('s'):
    cv.imwrite("Random_objects_disparity_map.png", disparity_colorMap)

cv.destroyAllWindows()




#==========DISPARITY TO DEPTH ESTIMATION==========



def disparityToDepth (baseline, doffs, focal_length, disparity):
    if disparity + doffs <= 0:  # division by zero error
        return None

    depth = (focal_length * baseline) / (disparity + doffs)

    return depth


# Pick a pixel
x = 300
y = 250

# Get disparity at that pixel
d = disparity_map[y, x]

# StereoBM stores disparity scaled by 16
d = d / 16.0

# Camera parameters
focal_length = 3979.911
baseline = 193.001
doffs = 124.343
Fx = 3979.911

#resizing the horizontal focal length
focal_length = Fx * (600/2964)

# Calculate depth
depth = disparityToDepth(
    baseline,
    doffs,
    focal_length,
    d
)
depth_in_meters = depth /1000

print("Pixel:", (x, y))
print("Disparity:", d)

if depth is not None:
    print("Estimated depth:", depth, "mm")
    print("Estimated depth:", depth_in_meters,"m")

else:
    print("Invalid disparity at this pixel.")

