import cv2 as cv
import os
import numpy as np

PATH = "~/crops/crops/"


# the calibration is camera-specific but very similar between cameras. You have to indicate which camera you wanna undistort
camera = 3

# see diagram below for camera numbering

#   TOP-DOWN VIEW OF THE VUZE LENSES
#
#     1 2
#   ==0=0==
#   |     |
# 8 0     0 3
#   |     |
# 7 0     0 4
#   |     |
#   ==0=0==
#     6 5
#      ^-------charging port

# camera numbering is different on camera and in calibration - camera starts at 1 and calibration starts at 0
extrinsics = np.load(f"extrinsics-{camera-1}.npz")

# load a random image
img = cv.imread(
    os.path.join(os.path.expanduser(PATH), f"camera_{camera}_frame_001752.png"))

# calculate the transformation matrix of how the pixels from the old image relate to the undistorted image
# THIS NEEDS TO BE DONE ONCE PER CAMERA, not per image
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(extrinsics["mtx"],
                                                 extrinsics["dist"], (w, h), 1,
                                                 (w, h))

# apply the undistortion matrix to the image
img2 = cv.undistort(img, extrinsics["mtx"], extrinsics["dist"], None,
                    newcameramtx)

# crop the image to contain only the coherent non-black parts
x, y, w, h = roi
img2_cropped = img2[y:y + h, x:x + w]

# show the sample
cv.imshow("original", cv.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2))))
cv.imshow("uncropped", cv.resize(img2,(int(img2.shape[1]/2),int(img2.shape[0]/2))))
cv.imshow("cropped", cv.resize(img2_cropped,(int(img2_cropped.shape[1]/2),int(img2_cropped.shape[0]/2))))
cv.waitKey(0)
