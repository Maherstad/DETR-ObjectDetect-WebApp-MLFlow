# DETR-ObjectDetect-WebApp-MLFlow
![object_detection_test_gif]([https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif](https://jmp.sh/s/GaJXP3NOYIQrdb9KOQpf))



## Introduction
ObjectDetect-WebApp-MLFlow is a comprehensive solution integrating object detection using a HuggingFace model, MLflow for model management, a Streamlit-based web interface, and Docker for deployment. 

## Features
- Object detection using DETR (Object Detection with transformers).
- Model management with MLflow.
- Streamlit-based front-end for user interaction.
- Docker support for both local and cloud deployment.
- Custom PyFunc class for MLflow vision models.


## Installation (local)
0. clone the repository  
```bash
git clone https://github.com/Maherstad/DETR-ObjectDetect-WebApp-MLFlow.git
```

1. create a virtual environment (using conda)  
```bash
conda create --name venv python=3.10 
conda activate venv
pip install -r requirements.txt
```
(optional) you can add a kernel from the virtual environment to your jupyter notebook using   
`python -m ipykernel install --user --name venv --display-name "venv kernel"`


2. navigate to the repository's folder and run the mlflow server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 6000
```

3. run the model.py to allow mlflow to log the model, register it and move it to Production
```bash
python object_detection_model.py
```

4. option-1 , locally : serve the model using mlflow so requests can be sent to it
```bash
mlflow models serve -m <path_to_model> -p 7000 --no-conda
```

5. launch the webpage
```bash
streamlit run webpage.py --server.port 5000
```

6. send requests to the server interactivally using the website or by going to `send_request_to_serve.ipynb` and running the code snippet

## Installation (cloud)

follow the steps 1. to 3. 

4. TODO : instruction for building Docker image and deployment
```bash
code
```
