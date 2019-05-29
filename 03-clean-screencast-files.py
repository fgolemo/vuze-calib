import numpy as np
import os
import matplotlib.pyplot as plt
from functools import reduce

from tqdm import tqdm

PATH = os.path.expanduser("~/Documents/tl1")

files = os.listdir(PATH)
files.sort()

good = 0
removed = 0

for f in tqdm(files):
    if f[-4:] == ".jpg":
        file_path = os.path.join(PATH, f)
        img = plt.imread(file_path)
        total_pixels = reduce(lambda x, y: x * y, img.shape)

        blacks = total_pixels - np.count_nonzero(img)
        whites = total_pixels - np.count_nonzero(img - 255)
        # print("whites: {},\tblacks: {}".format(whites, blacks))

        if whites == total_pixels:
            os.remove(file_path)
            print("removed", f)
            removed += 1

        else:
            good += 1

print("Good: {}, removed: {}".format(good, removed))
