from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from ultralytics import YOLO
from deepface import DeepFace
import os
import uuid
import cv2

app = Flask(__name__)
app.secret_key = "threatscan_secret_2024"

# Load trained YOLO model
model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/known_faces/<filename>")
def serve_known_face(filename):
    # Use absolute path to ensure the folder is found regardless of where the script is run from
    base_dir = os.path.dirname(os.path.abspath(__file__))
    known_faces_dir = os.path.join(base_dir, "known_faces")
    return send_from_directory(known_faces_dir, filename)

def process_detections(img, boxes):
    """Filters false positives, draws boxes manually, and returns valid detected classes."""
    detected_classes = []
    img_h, img_w = img.shape[:2]
    
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        w = x2 - x1
        h = y2 - y1
        
        if cls_id == 0: # Gun class
            # The model is confusing COCO Class 0 (Person) for Gun.
            # Relaxed ratio to > 2.0 (very tall and skinny) and > 40% height to ensure we don't accidentally ignore vertical guns
            if (h / max(w, 1)) > 2.0 and h > (img_h * 0.40):
                continue
            # Ignore massive bounding boxes (> 60% of image area) instead of 40%
            if (w * h) > (img_h * img_w * 0.60):
                continue
                
        detected_classes.append(cls_id)
        
        # Draw box manually
        color = (0, 0, 255) if cls_id == 0 else (0, 165, 255)
        label = "Gun" if cls_id == 0 else "Knife"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return img, detected_classes

@app.route("/analyze", methods=["POST"])
def analyze():
    threat_label = None
    alert_color = "green"
    result_file = None
    is_video = False
    snapshot_files = []
    original_filename = ""
    identified_person = "Unknown"
    match_image_file = None
    criminal_record_found = False

    file = request.files.get("file")

    if file and file.filename != "":
        original_filename = file.filename
        ext = file.filename.split(".")[-1].lower()
        filename = str(uuid.uuid4()) + "." + ext
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        # ================= IMAGE PROCESSING =================
        if ext in ["jpg", "jpeg", "png"]:
            # --- FACE IDENTIFICATION ---
            try:
                # find face in the db_path (criminal records)
                dfs = DeepFace.find(img_path=upload_path, db_path="known_faces", enforce_detection=False, silent=True)
                if len(dfs) > 0 and len(dfs[0]) > 0:
                    best_match = dfs[0].iloc[0]["identity"]
                    # e.g., "known_faces/MONICA.jpg" -> "MONICA"
                    identified_person = os.path.basename(best_match).split('.')[0]
                    match_image_file = os.path.basename(best_match)
                    criminal_record_found = True
            except Exception as e:
                print(f"Face ID error (Image): {e}")

            # Lowered confidence to 0.20 to catch weak model detections
            results = model(upload_path, conf=0.20)
            orig_img = results[0].orig_img.copy()
            result_img, detected_ids = process_detections(orig_img, results[0].boxes)

            if 0 in detected_ids:
                threat_label = "HIGH THREAT: Gun Detected"
                alert_color = "red"
            elif 1 in detected_ids:
                threat_label = "MEDIUM THREAT: Knife Detected"
                alert_color = "orange"
            else:
                threat_label = "SAFE: No Weapon Detected"
                alert_color = "green"

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
            first_frame_analyzed = False
            last_snapshot_frame = -9999
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                # --- FACE IDENTIFICATION (Check every 30 frames until a match is found) ---
                if identified_person == "Unknown" and frame_count % 30 == 0:
                    try:
                        temp_face_path = os.path.join(STATIC_FOLDER, "temp_face.jpg")
                        cv2.imwrite(temp_face_path, frame)
                        
                        dfs = DeepFace.find(img_path=temp_face_path, db_path="known_faces", enforce_detection=False, silent=True)
                        if len(dfs) > 0 and len(dfs[0]) > 0:
                            best_match = dfs[0].iloc[0]["identity"]
                            identified_person = os.path.basename(best_match).split('.')[0]
                            match_image_file = os.path.basename(best_match)
                            criminal_record_found = True
                    except Exception as e:
                        print(f"Face ID error (Video at frame {frame_count}): {e}")

                # Lowered confidence to 0.20 to catch weak model detections
                results = model(frame, conf=0.20)
                annotated_frame, detected_ids = process_detections(frame.copy(), results[0].boxes)

                # Check for Gun (0) or Knife (1) and save up to 5 distinct snapshots
                weapon_detected = 0 in detected_ids or 1 in detected_ids
                if weapon_detected and len(snapshot_files) < 5:
                    # Require at least 15 frames between snapshots so they aren't identical
                    if frame_count - last_snapshot_frame >= 15:
                        snapshot_name = "snapshot_" + str(uuid.uuid4()) + ".jpg"
                        snapshot_path = os.path.join(STATIC_FOLDER, snapshot_name)
                        cv2.imwrite(snapshot_path, annotated_frame)
                        snapshot_files.append(snapshot_name)
                        last_snapshot_frame = frame_count

                for cid in detected_ids:
                    detected_ids_total.add(cid)

                out.write(annotated_frame)

            cap.release()
            out.release()

            if 0 in detected_ids_total:
                threat_label = "HIGH THREAT: Gun Detected in Video"
                alert_color = "red"
            elif 1 in detected_ids_total:
                threat_label = "MEDIUM THREAT: Knife Detected in Video"
                alert_color = "orange"
            else:
                threat_label = "SAFE: No Weapon Detected in Video"
                alert_color = "green"

            result_file = output_name

    # Store result in session and redirect to results page
    session["result"] = {
        "result_file": result_file,
        "threat": threat_label,
        "color": alert_color,
        "is_video": is_video,
        "snapshot_files": snapshot_files,
        "original_filename": original_filename,
        "identified_person": identified_person,
        "match_image_file": match_image_file,
        "criminal_record_found": criminal_record_found,
    }
    return redirect(url_for("results"))


@app.route("/results")
def results():
    data = session.get("result", {})
    return render_template(
        "result.html",
        result_file=data.get("result_file"),
        threat=data.get("threat"),
        color=data.get("color", "green"),
        is_video=data.get("is_video", False),
        snapshot_files=data.get("snapshot_files", []),
        original_filename=data.get("original_filename", ""),
        identified_person=data.get("identified_person", "Unknown"),
        match_image_file=data.get("match_image_file"),
        criminal_record_found=data.get("criminal_record_found", False),
    )


if __name__ == "__main__":
    app.run(debug=True)