"""Call the model server from Python. Usage: python examples/client.py assets/street.jpg"""

import base64
import json
import os
import sys

import requests

path = sys.argv[1] if len(sys.argv) > 1 else "assets/street.jpg"
with open(path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:7100") + "/invocations",
    json={"dataframe_records": [{"image": image_b64}], "params": {"threshold": 0.7}},
    timeout=60,
)
response.raise_for_status()
print(json.dumps(response.json()["predictions"][0], indent=2))
