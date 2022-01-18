import time
import requests
from tqdm import tqdm
url = "https://europe-west1-mlops-course-337909.cloudfunctions.net/iris-function"
payload = {"input_data": "1, 2, 3, 4"}

for _ in tqdm(range(1000)):
    r = requests.get(url, params=payload)
    # print(r.content)

