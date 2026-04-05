from ultralytics import YOLO
import os

# Load trained model
model = YOLO("best.pt")

# Folder containing unseen images
image_folder = "test_images"

# Run prediction
results = model.predict(
    source=image_folder,
    imgsz=640,
    conf=0.25,
    save=True,
    project="results",
    name="unseen_predictions"
)

print("\nInference completed!")
print("Results saved in: results/unseen_predictions")