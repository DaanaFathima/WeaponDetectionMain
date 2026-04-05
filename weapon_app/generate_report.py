from ultralytics import YOLO
import pandas as pd
import os

model = YOLO("best.pt")

results = model.predict(
    source="test_images",
    conf=0.25
)

data = []

for r in results:

    image_name = os.path.basename(r.path)

    if r.boxes is None or len(r.boxes) == 0:

        data.append({
            "image": image_name,
            "class": "None",
            "confidence": 0
        })

    else:

        for box in r.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            data.append({
                "image": image_name,
                "class": model.names[cls],
                "confidence": round(conf,3)
            })

df = pd.DataFrame(data)

os.makedirs("results",exist_ok=True)

df.to_csv("results/prediction_report.csv",index=False)

print("Report saved to results/prediction_report.csv")