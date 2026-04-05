import os
import requests

folder = "test_images"
os.makedirs(folder, exist_ok=True)

def download_images(keyword, count, start_index):

    for i in range(count):

        url = f"https://source.unsplash.com/800x600/?{keyword}"

        img_data = requests.get(url).content

        filename = os.path.join(folder, f"test_{start_index+i+1}.jpg")

        with open(filename, "wb") as f:
            f.write(img_data)

        print("Downloaded:", filename)


print("Downloading person with knife images...")
download_images("person with knife",10,0)

print("Downloading person with gun images...")
download_images("person with gun",10,10)

print("Downloading person without weapon images...")
download_images("person portrait",5,20)

print("\n25 test images downloaded successfully!")