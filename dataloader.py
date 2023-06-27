import os
import imageio as io
import numpy as np
import cv2
import albumentations as A
import tensorflow as tf

# Dataset structure of bagls archive
folder = "training_224x224/"
ext_mask = "seg"
filetype = ".png"

def load_data(path_ds, aug_dataset=True):
    np.random.seed(42)
    img = []
    mask = []  

    # Dataset BAGLS - 1 folder with all images and masks
    files_dataset = os.listdir(path_ds / folder)
    image_number = [f_name for f_name in files_dataset if ext_mask not in f_name]
    
    #for x in range(len(image_number)):
    for x in range(20000):
        mask_file = io.imread(path_ds / folder / f"{x}_{ext_mask}{filetype}") / 255.
        mask.append(mask_file)
        im_file = io.imread(path_ds / folder / f"{x}{filetype}")

        if len(im_file.shape) == 3:
            im_file = cv2.cvtColor(im_file, cv2.COLOR_RGB2GRAY)
        img.append(im_file / 255.)
    
    img_arr = np.array([tf.expand_dims(i, -1) for i in np.array(img)])
    mask_arr = np.array([tf.expand_dims(i, -1) for i in np.array(mask)])

    # Shuffle images and masks in same order
    indices = np.arange(len(img))
    rand = indices
    np.random.shuffle(rand)
    img_arr = img_arr[rand]
    mask_arr = mask_arr[rand]

    # Dataset split if testset not explicitly given
    tr_ind = indices[0 : int(0.7 * len(img))]
    val_ind = indices[int(0.7 * len(img)) : int(0.8 * len(img))]
    ts_ind = indices[int(0.8 * len(img)) :]
    tr_img, tr_mask = img_arr[tr_ind], mask_arr[tr_ind]
    val_img, val_mask = img_arr[val_ind], mask_arr[val_ind]
    ts_img = img_arr[ts_ind]

    if aug_dataset:
       tr_img, tr_mask = augment_dataset(tr_img, tr_mask)

    return tr_img, tr_mask, val_img, val_mask, ts_img

def augment_dataset(tr_img, tr_mask):
    # Add augmentation to train dataset
    transform = A.Compose(
        [A.HorizontalFlip(p=0.5), A.Rotate(limit=10)]
    )
    x_img = []
    x_mask = []
    for i in range(len(tr_img)):
        transformed_train = transform(image=tr_img[i], mask=tr_mask[i])
        x_img.append(np.array(transformed_train["image"]))
        x_mask.append(np.array(transformed_train["mask"]))
    tr_img = np.array(x_img)
    tr_mask = np.array(x_mask)
    
    return tr_img, tr_mask
