import io
import base64
import json
import torch
import mlflow 

import pandas as pd
from PIL import Image,ImageDraw
from transformers import DetrImageProcessor, DetrForObjectDetection

from utils import decode_and_resize_image,json_to_image,image_to_json,dataframe_to_image,image_to_dataframe

# Custom PyFunc model
class DETRWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self,
                  tracking_uri = "http://0.0.0.0:6000",
                  set_experiment="Object Detection Experiment",
                  artifact_path = "object_detector",
                  registered_model_name = "DETR_Object_Detection_Model",
                ):
        
        self.tracking_uri = tracking_uri
        self.set_experiment =set_experiment
        self.artifact_path =artifact_path
        self.registered_model_name =registered_model_name


    def load_context(self,context):
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def log_model(self):
        
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.set_experiment)

            # Start an MLflow run
            with mlflow.start_run():

                # Log the custom PyFunc model
                model_info = mlflow.pyfunc.log_model(
                    artifact_path=self.artifact_path,
                    python_model=DETRWrapper(),
                    registered_model_name=self.registered_model_name
                )
            print("model logged successfully")

        except Exception as e:
            print(f"Error in logging model{e}")
            return None


    def register_and_stage_model(self,):
        run_id = os.environ['RUN_ID']
        subpath = "experiment_x"
        model_name = self.registered_model_name
        run_uri = f"runs://{run_id}/{subpath}"
        model_version = mlflow.register_model(run_uri,model_name)

        client = mlflow.MlflowClient()

        client.transition_model_version_stage(
                name = self.registered_model_name,
                version = 1,
            stage = "Staging",

            
        )
        
        return None

    def predict(self,context, input):

        """
        Generate predictions for the data.

        :param input: pandas.DataFrame with one column containing images to be scored. The image
                     column must contain base64 encoded binary content of the image files. The image
                     format must be supported by PIL (e.g. jpeg or png).

        :return: TODO fill return type
        """
        print("_"*25,"type of input data: ",type(input),"_"*25)
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

                # Get class name and draw it
                class_name = self.model.config.id2label[label.item()]
                text_position = (box[2], box[1])  # Top right corner of the bounding box
                draw.text(text_position, class_name, fill="red")
                
                # Optionally, you can print the detections as well
                # print(
                #     f"Detected {self.model.config.id2label[label.item()]} with confidence "
                #     f"{round(score.item(), 3)} at location {box}"
                # )



            return image_to_dataframe(image)

        except Exception as e:
            print(f"Error in processing input: {e}")
            return None


if __name__ == '__main__':
    detrwrapper = DETRWrapper()
    detrwrapper.log_model()
