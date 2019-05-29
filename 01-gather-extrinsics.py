import os
import numpy as np
import cv2 as cv
from tqdm import tqdm
import pickle

PATH = "~/crops/crops/"

files = os.listdir(os.path.expanduser(PATH))

files.sort()

# print (files[:3])
# camera_1_frame_000743.png

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = {}  # 3d point in real world space
imgpoints = {}  # 2d points in image plane.

for i in range(8):
    objpoints[i] = []
    imgpoints[i] = []

for idx, f in enumerate(tqdm(files)):
    if f[-4:] != ".png":
        continue

    components = f.split(".")
    components = components[0].split("_")
    cam_idx = int(components[1]) - 1
    # ['camera', '1', 'frame', '000743']

    img = cv.imread(
        os.path.join(os.path.expanduser(PATH), f))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 5), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7, 5), corners2, ret)
        cv.imshow('img', cv.resize(img, (500, 800)))
        key = cv.waitKey(1)
        # if key == ord(' '):
        objpoints[cam_idx].append(objp)
        imgpoints[cam_idx].append(corners)

    # if idx == 10:
    #     print (objpoints)
    #     print (imgpoints)

    if idx % 100 == 0:
        pickle.dump(objpoints, open("objpoints.pkl", "wb"))
        pickle.dump(imgpoints, open("imgpoints.pkl", "wb"))

cv.destroyAllWindows()

pickle.dump(objpoints, open("objpoints.pkl", "wb"))
pickle.dump(imgpoints, open("imgpoints.pkl", "wb"))


# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# img = cv.imread(os.path.join(os.path.expanduser(PATH), "camera_1_frame_000748.png"))
# h,  w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
#
# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv.imshow('img', cv.resize(dst,(500,800)))
# cv.waitKey(0)
