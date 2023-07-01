from unet import UNet
import json
from dataloader import load_data
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from segmentation_models.losses import dice_loss
from segmentation_models.metrics import iou_score
import argparse
from datetime import datetime
from pathlib import Path


def trainSegmentation(X, y, Xval, yval, batch_size=64, optimizer="Adam", epochs=50, folder="", num_classes=1):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = UNet(classes=num_classes)
    model.compile(optimizer, dice_loss, metrics=[iou_score])

    model_checkpoint_callback = ModelCheckpoint(f"{folder}/bagls_chkpoint_{now}.h5",save_best_only=True)
    csv_callback = CSVLogger(f"{folder}/bagls_csv_{now}.csv")
    
    h = model.fit(X, 
              y, 
              validation_data=(Xval,yval),
              batch_size=batch_size, 
              epochs=epochs,
              callbacks=[csv_callback, model_checkpoint_callback])
    
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
    result_dict = {}

    X, y, Xval, yval, test = load_data(DIR_SOURCE)

    history = trainSegmentation(X, y, Xval, yval, batch_size=32, optimizer="Adam", epochs=EPOCHS, folder=DIR_OUTPUT, num_classes=1)
    
    result_dict["iou"] = history['val_iou_score']
    with open (f"{DIR_OUTPUT}/results.json", "w") as f:
        json.dump(result_dict, f)


