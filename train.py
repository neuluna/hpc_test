from unet import UNet
from dataloader import load_data
from datetime import datetime
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from segmentation_models.losses import dice_loss
from segmentation_models.metrics import iou_score
import argparse
from tqdm import tqdm
from pathlib import Path


def trainSegmentation(X, y, Xval, yval, batch_size=4, optimizer="Adam", epochs=5, folder="", num_classes=1):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = UNet(classes=num_classes)
    model.compile(optimizer, dice_loss, metrics=[iou_score])
    
    h = model.fit(X, 
              y, 
              validation_data=(Xval,yval),
              batch_size=batch_size, 
              epochs=epochs,
              verbose=0, 
              callbacks=[CSVLogger(str(folder.joinpath(f"{now}.csv")))])
    
    return max(h.history['val_iou_score'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', default ="/shares/ANKI/Projects/", help="The path of to dataset dir.")
    parser.add_argument('-o', '--output', default = "/output", help="The path for output.")
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-e', '--epochs', type=int, default=20)
    args = parser.parse_args()

    DIR_SOURCE = Path(args.source)
    DIR_OUTPUT = Path(args.output)

    X, y, Xval, yval = load_data(DIR_SOURCE)

    trainSegmentation(X, y, Xval, yval, batch_size=32, optimizer="Adam", epochs=20, folder="", num_classes=1)
