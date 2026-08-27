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

window_names = ["Motor0", "Motor1", "rand0", "rand1"]
for j, path in enumerate(image_paths):

    img = cv.imread(path)

    if img is None:
        print("Could not load:", path)
        continue

    resized = cv.resize(img, (width, height))

    gray_motor = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    window_name = window_names[j]

    cv.imshow(window_name, gray_motor)

    k = cv.waitKey(0)

    if k == ord('s'):
        cv.imwrite(window_name + ".png", gray_motor)
        print(f"{window_name}.png saved")

    cv.destroyAllWindows()