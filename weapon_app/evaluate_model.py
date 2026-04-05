import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load prediction report
pred = pd.read_csv("results/prediction_report.csv")

# Load ground truth
gt = pd.read_csv("ground_truth.csv")

predictions = []
truth = []

for i in range(len(gt)):

    image = gt.iloc[i]["image"]
    true_class = gt.iloc[i]["true_class"]

    rows = pred[pred["image"] == image]

    if len(rows) == 0:
        predicted = "None"
    else:
        predicted = rows.iloc[0]["class"]

    truth.append(true_class)
    predictions.append(predicted)

# Convert class labels
label_map = {
    "class_0": "Gun",
    "class_1": "Knife"
}

predictions = [label_map.get(x, x) for x in predictions]

# Count results
correct = sum([t == p for t,p in zip(truth,predictions)])
wrong = len(truth) - correct
no_detection = predictions.count("None")

print("\n===== MODEL EVALUATION =====\n")

print("Total images:",len(truth))
print("Correct predictions:",correct)
print("Wrong predictions:",wrong)
print("No detections:",no_detection)

accuracy = correct/len(truth)

print("\nAccuracy:",round(accuracy*100,2),"%")

# Classification report
print("\nClassification Report:\n")
print(classification_report(truth,predictions))

# Confusion matrix
labels = ["Gun","Knife","None"]

cm = confusion_matrix(truth,predictions,labels=labels)

plt.figure(figsize=(6,5))

sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("results/confusion_matrix.png")

plt.show()