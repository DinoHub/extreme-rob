import cv2
import numpy as np
import random
from PIL import Image
import os
import copy
# import torchvision.transforms as transforms
from skimage.util import random_noise
import imagecorruptions as ic


source_dir = "/home/vishesh/Desktop/datasets/ships-data/X_true_val_240_320"
target_dir = "/home/vishesh/Desktop/datasets/ships-data/perturbed_test/varying"

def add_noise(img_arr):
    mode = random.choice(["gaussian", "localvar", "poisson", "salt", "pepper", "s&p", "speckle"])
    noise_img = random_noise(img_arr, mode=mode)
    return noise_img

# def add_affine(image_path):
#     img = Image.open(image_path)

#     max_degree = 10
#     max_translate = (0.3,0.3)
#     scale_range = (0.7,1.3)
#     shear_range = (10,10,10,10)
 
#     # define an transform
#     transform = transforms.RandomAffine(degrees=max_degree, translate=max_translate, 
#                                         scale=scale_range, shear=shear_range)
    
#     # apply the above transform on image
#     img = transform(img)
    
#     img.save("new_image.png")

def add_contrast_or_brightness(img_arr, mode="contrast"):
    b, g, r = cv2.split(img_arr)

    if mode == "contrast":
        alpha, beta = random.choice([2,3,4,5]), 0
    elif mode == "brightness":
        alpha, beta = 1, random.choice([-200,-100,100,200])

    b_adjusted = cv2.convertScaleAbs(b, alpha=alpha, beta=beta)
    g_adjusted = cv2.convertScaleAbs(g, alpha=alpha, beta=beta)
    r_adjusted = cv2.convertScaleAbs(r, alpha=alpha, beta=beta) 

    adjusted_img = cv2.merge([b_adjusted, g_adjusted, r_adjusted])
    return adjusted_img

def add_corruption(img_arr):
    for corruption in ic.get_corruption_names():
        severity = random.randrange(1,5)
        corrupted = ic.corrupt(img_arr, corruption_name=corruption, severity=severity)
        return corrupted


def add_distortion(image_filename):
    image_path = os.path.join(source_dir, image_filename)

def add_padding(img_arr):
    h, w = img_arr.shape[:2]

    top = int(random.uniform(0.05,0.15) * img_arr.shape[0])
    bottom = top
    left = int(random.uniform(0.05,0.15) * img_arr.shape[1])
    right = left

    borderType = random.choice([cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP])
    value = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] if borderType == cv2.BORDER_CONSTANT else None

    dst = cv2.copyMakeBorder(img_arr, top, bottom, left, right, borderType=borderType, value=value)
    dst = cv2.resize(dst, (w,h))
    return dst


def add_patch(image_filename):
    image_path = os.path.join(source_dir, image_filename)


for j, _ in enumerate(os.listdir(source_dir), start=1):
    print(j)
    np_filename = f"{j}.npy"
    np_filepath = os.path.join(source_dir, np_filename)

    img_arrs = np.load(np_filepath)
    img_arrs_copy = copy.deepcopy(img_arrs)

    img_arrs = (img_arrs * 255).astype(np.uint8)

    for corruption in ["shot_noise"]: #ic.get_corruption_names():
        os.mkdir(os.path.join(target_dir, corruption))
        print(corruption)
        # if corruption == "glass_blur":
        #     continue
        for severity in range(1,11):
            os.mkdir(os.path.join(target_dir, corruption, str(severity)))
            for i in range(img_arrs.shape[0]):
                img_arr = img_arrs[i]

                perturbed = ic.corrupt(img_arr, corruption_name=corruption, severity=severity)

                img_arrs_copy[i] = perturbed / 255

            np.save(os.path.join(target_dir, corruption, str(severity), np_filename), img_arrs_copy)

    print("done")
    break

    




