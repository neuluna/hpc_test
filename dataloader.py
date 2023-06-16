import os
import sys
import pathlib
import json
import imageio as io
import numpy as np
import cv2
import albumentations as A
from tqdm import trange
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image


def load_data(self, path_share, test_scenario=True):
    np.random.seed(42)
    img = []
    mask = []  
    if self.dataset == "bagls2":
        print("Datagenerator: ",self.dataset)
        # Dataset like BAGLS - 1 folder with all images and masks
        files_dataset = os.listdir(path_share / self.path)
        # print(files_dataset)
        image_number = [f_name for f_name in files_dataset if self.ext_mask not in f_name]
        print(len(image_number))
        
        #for x in range(len(image_number)):
        for x in range(100):
            # Maske
            mask_file = io.imread(path_share / self.path / f"{x}_{self.ext_mask}{self.filetype}") / 255.
            mask.append(mask_file)

            # Bild
            im_file = io.imread(path_share / self.path / f"{x}{self.filetype}")

            if len(im_file.shape) == 3:
                im_file = cv2.cvtColor(im_file, cv2.COLOR_RGB2GRAY)
            img.append(im_file / 255.)
        

        img_arr = np.array([tf.expand_dims(i, -1) for i in np.array(img)])
        mask_arr = np.array([tf.expand_dims(i, -1) for i in np.array(mask)])


        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))
        ax1.imshow(img_arr[0,:,:,:])
        ax2.imshow(mask_arr[0,:,:,:])
        plt.savefig(f"output/{self.dataset}_0.png")
    
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


    return tr_img, tr_mask, val_img, val_mask, ts_img

    