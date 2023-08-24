"""
This code converts the numpy ships dataset into png images
"""



import copy
import os
import numpy as np
import cv2

source_dir = "/home/vishesh/Desktop/datasets/ships-data/perturbed_test/varying/shot_noise"
target_dir = "/home/vishesh/Desktop/datasets/ships-data/test_images/varying/shot_noise"
os.mkdir(target_dir)

for severity in range(1,11):
    os.mkdir(os.path.join(target_dir, f"{severity}"))
    counter = 1
    for num in range(1,16):
        np_filename = f"{num}.npy"
        np_filepath = os.path.join(source_dir, f"{severity}", np_filename)

        img_arrs = np.load(np_filepath)
        img_arrs = (img_arrs * 255).astype(np.uint8)

        for i in range(img_arrs.shape[0]):
            cv2.imwrite(os.path.join(target_dir, f"{severity}", f"img{counter}.png"), img_arrs[i])
            counter += 1

        break

