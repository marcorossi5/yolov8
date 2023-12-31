import argparse
from pathlib import Path

from ultralytics import YOLO

def main(dataset_folder: Path, epochs: int):
    # Load a pretrained YOLO model (recommended for training)
    yolo_checkpoint = 'models/yolov8n.pt'
    model = YOLO(yolo_checkpoint)

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    if epochs is not None:
        results = model.train(data=dataset_folder, epochs=epochs)

    # Evaluate the model's performance on the validation set
    results = model.val(data=dataset_folder)

    # # Export the model to ONNX format
    success = model.export()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=Path, required=True, help="dataset folder path")
    parser.add_argument("-e", "--epochs", type=int, default=None, help="number of epochs to finetune the model")
    args = parser.parse_args()
    main(args.dataset.absolute(), args.epochs)