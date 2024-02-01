import json
import requests
import base64
import pandas as pd
from utils import string_to_pil_image


path = './assets/bedroom.jpg'
host = 'http://127.0.0.1'
port = '7000'

def read_image(x):
    with open(x, "rb") as f:
        return f.read()

path = './assets/bedroom.jpg'
data = {"image": [base64.encodebytes(read_image(path)).decode()]}

data = pd.DataFrame(data).to_csv(index=False) 

response = requests.post(
    url=f"{host}:{port}/invocations",
    data=data,  # Sending the CSV data
    headers={"Content-Type": "text/csv"},  # Updated content type to 'text/csv'
)

if response.status_code != 200:
    raise Exception(f"Status Code {response.status_code}. {response.text}")

#######################

x = json.loads(response.text)
from utils import decode_and_resize_image
string_to_pil_image(x['predictions'][0]['0'])



#request to text generation server
#url = "http://127.0.0.1:5000/invocations"
#headers = {
#    "Content-Type": "application/json"
#}
#data = {
#    "inputs": ["test this is a test"]
#}
#response = requests.post(url, headers=headers, data=json.dumps(data))
#
#print(response.text)