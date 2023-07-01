# Example of the implementation of a U-net.

Implementation of a U-net that can be trained on different datasets.

## Project description

This repository contains an example of segmentation training in Tensorflow/Keras. It contains a dataloader, a Unet implementation, a training function, and two shell scripts to get the training running.

## Table of Contents

- [Dataloader](#Dataloader)
- [U-Net](#U-Net)
- [Training](#Training)
- [HPC-Job](#HPC-Job)


## Dataloader

def load_data(path_ds, aug_dataset=True) - loads the images and mask files or their data structure. It splits the dataset into train, validation and test set. The aug_dataset variable defines whether augmentation should be applied to the training dataset or not.

Due to OOM issues, only 20,000 images are currently loaded.

## U-Net

Implementation of a U-Net with filter size 16, 4 layers and InstanceNormalization.

## Training

Train.py takes four input parameters and batchsize is set to 64.
* -s - path to the dataset
* -o - path where the output should be stored
* -d - name of the dataset
* -e - number of epochs


## HPC job

* hpc-job.sh - defines all hpc requirements at the beginning, creates environment variables, calls the conda environment and all required packages and executes the script for training
* hpc-submit.sh - calls the hpc job, so the script can be started multiple times on the HPC