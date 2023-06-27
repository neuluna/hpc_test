from unet import UNet
from dataloader import load_data
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from segmentation_models.losses import dice_loss
from segmentation_models.metrics import iou_score
import argparse
from tqdm import tqdm
from pathlib import Path


def trainSegmentation(X, y, Xval, yval, batch_size=4, optimizer="Adam", epochs=5, num_classes=1):
    model = UNet(classes=num_classes)
    model.compile(optimizer, dice_loss, metrics=[iou_score])
    
    h = model.fit(X, 
              y, 
              validation_data=(Xval,yval),
              batch_size=batch_size, 
              epochs=epochs)
    
    return h.history
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', default ="/shares/ANKI/Projects/", help="The path of to dataset dir.")
    parser.add_argument('-o', '--output', default = "/output", help="The path for output.")
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-e', '--epochs', type=int, default=20)
    args = parser.parse_args()

    DIR_SOURCE = Path(args.source)
    DIR_OUTPUT = Path(args.output)
    DATASET= args.dataset
    EPOCHS = args.epochs

    X, y, Xval, yval, test = load_data(DIR_SOURCE)

    history = trainSegmentation(X, y, Xval, yval, batch_size=32, optimizer="Adam", epochs=EPOCHS, num_classes=1)
    print(history['val_iou_score'])
