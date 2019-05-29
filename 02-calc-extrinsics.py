import pickle
import cv2 as cv
import os
import numpy as np
from tqdm import trange

objpoints = pickle.load(open("objpoints.pkl", "rb"))
imgpoints = pickle.load(open("imgpoints.pkl", "rb"))

PATH = "~/crops/crops/"

for i in range(8):
    print(i, len(imgpoints[i]), len(objpoints[i]))

img = cv.imread(
    os.path.join(os.path.expanduser(PATH), "camera_1_frame_000743.png"))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

for i in trange(8):
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints[i],
                                                      imgpoints[i],
                                                      gray.shape[::-1], None,
                                                      None)
    np.savez(
        "extrinsics-{}.npz".format(i),
        ret=ret,
        mtx=mtx,
        dist=dist,
        rvecs=rvecs,
        tvecs=tvecs)
    quit()

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
