from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
import cv2

app = Flask(__name__)

# Load trained YOLO model
model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    threat_label = None
    alert_color = "green"
    result_file = None
    is_video = False
    snapshot_file = None  # NEW: snapshot image

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            ext = file.filename.split(".")[-1].lower()
            filename = str(uuid.uuid4()) + "." + ext
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            # ================= IMAGE PROCESSING =================
            if ext in ["jpg", "jpeg", "png"]:
                results = model(upload_path, conf=0.25)
                detected_ids = [int(box.cls[0]) for box in results[0].boxes]

                if 0 in detected_ids:
                    threat_label = "🚨 HIGH THREAT: Gun Detected 🔫"
                    alert_color = "red"
                elif 1 in detected_ids:
                    threat_label = "⚠ MEDIUM THREAT: Knife Detected 🔪"
                    alert_color = "orange"
                else:
                    threat_label = "✅ SAFE: No Weapon Detected"
                    alert_color = "green"

                result_img = results[0].plot()
                result_path = os.path.join(STATIC_FOLDER, filename)
                cv2.imwrite(result_path, result_img)

                result_file = filename

            # ================= VIDEO PROCESSING =================
            elif ext in ["mp4", "avi", "mov"]:
                is_video = True

                cap = cv2.VideoCapture(upload_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                output_name = str(uuid.uuid4()) + ".mp4"
                output_path = os.path.join(STATIC_FOLDER, output_name)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                detected_ids_total = set()
                snapshot_saved = False

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, conf=0.25)

                    detected_ids = [int(box.cls[0]) for box in results[0].boxes]

                    # Save snapshot ONLY first time gun is detected
                    if 0 in detected_ids and not snapshot_saved:
                        snapshot_name = "snapshot_" + str(uuid.uuid4()) + ".jpg"
                        snapshot_path = os.path.join(STATIC_FOLDER, snapshot_name)
                        cv2.imwrite(snapshot_path, results[0].plot())
                        snapshot_file = snapshot_name
                        snapshot_saved = True

                    for cid in detected_ids:
                        detected_ids_total.add(cid)

                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)

                cap.release()
                out.release()

                if 0 in detected_ids_total:
                    threat_label = "🚨 HIGH THREAT: Gun Detected in Video 🔫"
                    alert_color = "red"
                elif 1 in detected_ids_total:
                    threat_label = "⚠ MEDIUM THREAT: Knife Detected in Video 🔪"
                    alert_color = "orange"
                else:
                    threat_label = "✅ SAFE: No Weapon Detected in Video"
                    alert_color = "green"

                result_file = output_name

    return render_template(
        "index.html",
        result_file=result_file,
        threat=threat_label,
        color=alert_color,
        is_video=is_video,
        snapshot_file=snapshot_file
    )


if __name__ == "__main__":
    app.run(debug=True)