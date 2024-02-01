import io
import time 
import requests
import json
import mlflow 
import base64
import pandas as pd 
import requests
import json
import numpy as np 

from io import BytesIO
from PIL import Image, ImageDraw

def detect_objects(image): #input is bytes object 
    
    url = "http://127.0.0.1:7000/invocations"  # Replace with the actual URL of your model server
    headers={"Content-Type": "text/csv"}
    
    # Send the image data to the server
    data = {"image": [base64.encodebytes(image).decode()]}
    data = pd.DataFrame(data).to_csv(index=False)
    response = requests.post(url, headers=headers, data=data)

    # Assuming the response contains the modified image with predictions
    if response.status_code == 200:
        # Convert the response content back to a PIL Image
        
        x = json.loads(response.text)
        response_image = string_to_pil_image(x['predictions'][0]['0'])
        return response_image
    else:
        raise Exception("Failed to get response from the model server")


def get_image_bytes(img, format="PNG"):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()


def decode_and_resize_image(raw_bytes, size):
    """
    Read, decode and resize raw image bytes (e.g. raw content of a jpeg file).
    
    # input : bytes representation of an image
    # output : numpy array representation of the image

    :param raw_bytes: Image bits, e.g. jpeg image.
    :param size: requested output dimensions
    :return: Multidimensional numpy array representing the resized image.
    """
    return np.asarray(Image.open(BytesIO(raw_bytes)).resize(size), dtype=np.float32)


def json_to_image(json_string):
    # Convert JSON string to DataFrame
    df = pd.read_json(json_string, orient='split')
    # Extract the first row's image data (base64 encoded)
    encoded_image = df.iloc[0]['image']
    # Decode the base64 string to binary data
    image_data = base64.decodebytes(encoded_image.encode('utf-8'))
    # Convert binary data to PIL image
    image = Image.open(io.BytesIO(image_data))
    return image

def image_to_json(image):
    # Convert PIL image to binary data
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # You can change the format to PNG if needed
    image_binary = buffered.getvalue()
    # Encode the binary data to base64
    encoded_string = base64.encodebytes(image_binary).decode()
    # Create a DataFrame with the encoded string
    df = pd.DataFrame([encoded_string], columns=['image'])
    # Convert the DataFrame to a JSON string
    json_string = df.to_json(orient='split')
    return json_string


def dataframe_to_image(df):
    """
    Convert a pandas DataFrame with a single row containing a base64 encoded image to a PIL image.

    :param df: Pandas DataFrame with one row and one column containing a base64 encoded image.
    :return: PIL Image object.
    """
    # Extract the base64 image string
    base64_image = df.iloc[0, 0]  # Accessing the first (and only) element
    
    # Decode the base64 string
    image_data = base64.b64decode(base64_image)
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_data))
    
    return image



def image_to_dataframe(image):
    """
    Convert a PIL Image to a pandas DataFrame with a single row containing a base64 encoded image.

    :param image: PIL Image object.
    :return: Pandas DataFrame with one row and one column containing a base64 encoded image.
    """
    # Convert PIL Image to a byte stream
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # You can change the format if necessary
    img_byte = buffered.getvalue()

    # Encode byte stream to base64
    base64_image = base64.b64encode(img_byte).decode()

    # Create DataFrame
    df = pd.DataFrame([base64_image])

    return df

def string_to_pil_image(base64_string):
    """
    Convert a base64 encoded string representation of an image to a PIL Image.

    :param base64_string: Base64 encoded string of an image.
    :return: PIL Image object.
    """
    image_bytes = base64.b64decode(base64_string)  # Decode the base64 string
    image_stream = io.BytesIO(image_bytes)  # Create a BytesIO object from the decoded bytes
    image = Image.open(image_stream)  # Open image using PIL

    return image
