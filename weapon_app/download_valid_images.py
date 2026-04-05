import os
import requests
from PIL import Image
from io import BytesIO

folder = "test_images"
os.makedirs(folder, exist_ok=True)

keywords = [
    "person knife",
    "person knife",
    "person knife",
    "person knife",
    "person knife",
    "person knife",
    "person knife",
    "person knife",
    "person knife",
    "person knife",

    "person gun",
    "person gun",
    "person gun",
    "person gun",
    "person gun",
    "person gun",
    "person gun",
    "person gun",
    "person gun",
    "person gun",

    "person portrait",
    "person portrait",
    "person portrait",
    "person portrait",
    "person portrait"
]

print("Downloading valid test images...\n")

for i, keyword in enumerate(keywords):

    url = f"https://picsum.photos/seed/{keyword.replace(' ','')}{i}/800/600"

    response = requests.get(url)

    try:
        img = Image.open(BytesIO(response.content))
        path = os.path.join(folder, f"test_{i+1}.jpg")
        img.save(path)
        print("Saved:", path)

    except:
        print("Invalid image skipped")

print("\nImages downloaded successfully!")