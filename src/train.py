import os
from pathlib import Path

import mlflow
import torch
import yaml
from dotenv import load_dotenv
from torch import nn
from ultralytics import YOLO

from utils import save_metrics_and_params, save_model


load_dotenv()

MLFLOW_TRACKING_URI=os.getenv('MLFLOW_TRACKING_URI')

root_dir = Path(__file__).resolve().parents[1]  # root directory absolute path
data_dir = os.path.join(root_dir, "data/raw/wildfire-raw-yolov8")
data_yaml_path = os.path.join(data_dir, "data.yaml")
metrics_path = os.path.join(root_dir, 'reports/train_metrics.json')

def validate_model(model, val_loader, device, loss_criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            total_loss += loss.item() * images.size(0)
            total_correct += (predicted == labels).sum().item()
            total_samples += images.size(0)

    accuracy = total_correct / total_samples
    loss = total_loss / total_samples

    return accuracy, loss

if __name__ == '__main__':

    # load the configuration file 
    with open(r"params.yaml") as f:
        params = yaml.safe_load(f)

    # set the tracking uri 
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # start mlflow experiment 
    with mlflow.start_run(run_name=params['name']):
        # load a pre-trained model 
        pre_trained_model = YOLO(params['model_type'])
        # mlflow.end_run()
        # Start MLflow run
        mlflow.start_run()

        # train 
        model = pre_trained_model.train(
            data=data_yaml_path,
            imgsz=params['imgsz'],
            batch=params['batch'],
            epochs=params['epochs'],
            optimizer=params['optimizer'],
            lr0=params['lr0'],
            seed=params['seed'],
            pretrained=params['pretrained'],
            name=params['name']
        )

        # log params with mlflow
        mlflow.log_param('model_type', params['model_type'])
        mlflow.log_param('epochs',params['epochs'])
        mlflow.log_param('optimizer', params['optimizer'])
        mlflow.log_param('learning_rate', params['lr0'])

        # Log the trained model to MLflow
        # mlflow.pytorch.log_model(model, "model")


        # Validate the model
        # loss_criterion = nn.CrossEntropyLoss()

        # save model
        save_model(experiment_name=params['name']) 

        # save metrics csv file and training params 
        save_metrics_and_params(experiment_name=params['name'])

        mlflow.end_run()

         










