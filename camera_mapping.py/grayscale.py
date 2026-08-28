import cv2 as cv
import numpy as np

width = 600
height = 500

image_paths = [
    "camera_mapping.py\\dataset\\im0\\im0.png",
    "camera_mapping.py\\dataset\\im0\\im1.png",
    "camera_mapping.py\\dataset\\img1\\im0.png",
    "camera_mapping.py\\dataset\\img1\\im1.png"
]

window_names = ["FMotor0", "FMotor1", "Frand0", "Frand1"]
scale = 1
delta = 0
ddepth = cv.CV_16S
for j, path in enumerate(image_paths):

    img = cv.imread(path)

    if img is None:
        print("Could not load:", path)
        continue

    resized = cv.resize(img, (width, height))

    gray_img = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    # applied sobel filteration for edge detection
    filtered = cv.GaussianBlur(gray_img, (3, 3), 0)

    grad_x = cv.Sobel(filtered, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(filtered, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    
    
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    
    
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


    window_name = window_names[j]

    cv.imshow(window_name, grad)

    k = cv.waitKey(0)

    if k == ord('s'):
        cv.imwrite(window_name + ".png", grad)
        print(f"{window_name}.png saved")

    cv.destroyAllWindows()