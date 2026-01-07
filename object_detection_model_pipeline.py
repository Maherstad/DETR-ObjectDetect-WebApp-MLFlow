import os
import torch
import mlflow
import pandas as pd
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection

from utils import dataframe_to_image, image_to_dataframe

# Custom PyFunc model
class DETRWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self,
                 tracking_uri="http://127.0.0.1:5050",
                 set_experiment="Object Detection Experiment",
                 artifact_path="object_detector",
                 registered_model_name="DETR_Object_Detection_Model",
                ):
        
        self.tracking_uri = tracking_uri
        self.set_experiment = set_experiment
        self.artifact_path = artifact_path
        self.registered_model_name = registered_model_name

    def load_context(self, context):
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def log_model(self):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.set_experiment)

            # Start an MLflow run
            with mlflow.start_run() as run:
                # Log the custom PyFunc model using the current instance
                model_info = mlflow.pyfunc.log_model(
                    artifact_path=self.artifact_path,
                    python_model=self,
                    registered_model_name=self.registered_model_name
                )
            print("Model logged successfully. Run ID: ", run.info.run_id)
            return run.info.run_id  # Returning the run_id for further use

        except Exception as e:
            print(f"Error in logging model: {e}")
            return None

    def register_and_stage_model(self, run_id):
        if not run_id:
            print("Run ID not provided. Cannot register model.")
            return None

        try:
            model_uri = f"runs:/{run_id}/{self.artifact_path}"
            model_version_info = mlflow.register_model(model_uri, self.registered_model_name)

            client = mlflow.MlflowClient()
            client.transition_model_version_stage(
                name=self.registered_model_name,
                version=model_version_info.version,
                stage="Production"
            )
            print("Model registered and staged successfully.")
            return model_version_info

        except Exception as e:
            print(f"Error in registering and staging model: {e}")
            return None

    def predict(self, context, input):
        """
        Generate predictions for the data.
        :param input: pandas.DataFrame with one column containing images to be scored.
        :return: pandas.DataFrame with the processed image
        """
        print("_" * 25, "Type of input data: ", type(input), "_" * 25)
        try:
            image = dataframe_to_image(input)

            # Process the image and perform object detection
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)

            # Convert outputs to COCO API and filter detections
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(outputs,
                                                                  target_sizes=target_sizes,
                                                                  threshold=0.9)[0]

            # Draw the bounding boxes on the image
            draw = ImageDraw.Draw(image)
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                draw.rectangle(box, outline="red", width=3)

                class_name = self.model.config.id2label[label.item()]
                text_position = (box[2], box[1])  # Top right corner of the bounding box
                draw.text(text_position, class_name, fill="red")

            return image_to_dataframe(image)

        except Exception as e:
            print(f"Error in processing input: {e}")
            return None


if __name__ == '__main__':
    detrwrapper = DETRWrapper()
    run_id = detrwrapper.log_model()
    if run_id:
        detrwrapper.register_and_stage_model(run_id)
