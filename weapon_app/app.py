from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, Response
from ultralytics import YOLO
from deepface import DeepFace
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silence TF info/warnings
import torch
import uuid
import cv2

app = Flask(__name__)
app.secret_key = "threatscan_secret_2024"

# Load trained YOLO model with automatic GPU/CPU selection
device = 0 if torch.cuda.is_available() else 'cpu'
model = YOLO("best.pt")
model.to(device)
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Global state to track live camera session
live_session_data = {}


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
        area = w * h
        img_area = img_h * img_w
        
        # Gun (0) False Positive Filtering
        if cls_id == 0: 
            # The model is confusing COCO Class 0 (Person) for Gun.
            # Relaxed ratio to > 2.0 (very tall and skinny) and > 40% height to ensure we don't accidentally ignore vertical guns
            if (h / max(w, 1)) > 2.0 and h > (img_h * 0.40):
                continue
            # Ignore massive bounding boxes (> 40% of image area)
            if area > (img_area * 0.40):
                continue
                
        # Knife (1) False Positive Filtering
        if cls_id == 1:
            # False positives (like bedsheets) tend to be large, blocky squares.
            # Real knives in a hand are typically small and elongated.
            if area > (img_area * 0.15): # If it takes up >15% of the screen, it's not a knife.
                continue
                
            # Check aspect ratio (knives are long/thin, not perfectly square)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
            if aspect_ratio < 1.2: # If it's perfectly square, ignore it
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
            # Use imgsz=640 to match training and catch small weapons
            results = model(upload_path, conf=0.20, imgsz=640)
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
            last_boxes = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                # --- FACE IDENTIFICATION (Check every 30 frames until a match is found) ---
                if identified_person == "Unknown" and frame_count % 30 == 0:
                    try:
                        # Process frame directly in memory (no disk write)
                        dfs = DeepFace.find(img_path=frame, db_path="known_faces", enforce_detection=False, silent=True)
                        if len(dfs) > 0 and len(dfs[0]) > 0:
                            best_match = dfs[0].iloc[0]["identity"]
                            identified_person = os.path.basename(best_match).split('.')[0]
                            match_image_file = os.path.basename(best_match)
                            criminal_record_found = True
                    except Exception as e:
                        print(f"Face ID error (Video at frame {frame_count}): {e}")

                # --- AI FRAME SKIPPING (Speed up by 3x) ---
                # Run YOLO model only every 3rd frame to save massive processing time.
                if frame_count % 3 == 0 or last_boxes is None:
                    results = model(frame, conf=0.30, imgsz=640, verbose=False)
                    last_boxes = results[0].boxes

                annotated_frame, detected_ids = process_detections(frame.copy(), last_boxes)

                # Check for Gun (0) or Knife (1) and save distinct snapshots
                weapon_detected = 0 in detected_ids or 1 in detected_ids
                if weapon_detected:
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

def gen_frames():
    """Generator function for live camera feed with weapon & suspect detection."""
    global live_session_data
    cap = cv2.VideoCapture(0)
    frame_count = 0
    identified_person = "Unknown"
    last_snapshot_frame = -9999

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1

        # Suspect Identification (run every 30 frames for performance)
        if identified_person == "Unknown" and frame_count % 30 == 0:
            try:
                # Need to write frame to disk temporarily for deepface if img_path=frame fails, 
                # but DeepFace supports numpy array in newer versions.
                dfs = DeepFace.find(img_path=frame, db_path="known_faces", enforce_detection=False, silent=True)
                if len(dfs) > 0 and len(dfs[0]) > 0:
                    best_match = dfs[0].iloc[0]["identity"]
                    identified_person = os.path.basename(best_match).split('.')[0]
                    live_session_data["identified_person"] = identified_person
                    live_session_data["criminal_record_found"] = True
            except Exception as e:
                pass

        # Weapon Detection
        results = model(frame, conf=0.30, imgsz=640, verbose=False)
        annotated_frame, detected_ids = process_detections(frame.copy(), results[0].boxes)

        # Update threat snapshot and state
        weapon_detected = 0 in detected_ids or 1 in detected_ids
        if weapon_detected:
            # Space out snapshots by at least 15 frames
            if frame_count - last_snapshot_frame >= 15:
                snapshot_name = "snapshot_live_" + str(uuid.uuid4()) + ".jpg"
                snapshot_path = os.path.join(STATIC_FOLDER, snapshot_name)
                cv2.imwrite(snapshot_path, annotated_frame)
                live_session_data["snapshot_files"].append(snapshot_name)
                last_snapshot_frame = frame_count

            if 0 in detected_ids:
                live_session_data["threat"] = "HIGH THREAT: Gun Detected in Live Camera"
                live_session_data["color"] = "red"
            elif 1 in detected_ids and live_session_data["color"] != "red":
                live_session_data["threat"] = "MEDIUM THREAT: Knife Detected in Live Camera"
                live_session_data["color"] = "orange"

        # Overlay Suspect Name if identified
        if identified_person != "Unknown":
            cv2.putText(annotated_frame, f"SUSPECT: {identified_person}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route that serves the MJPEG stream for the live camera."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def live():
    """Route that renders the live camera UI page."""
    global live_session_data
    # Reset session data for the new live session
    live_session_data = {
        "result_file": None,
        "threat": "SAFE: No Weapon Detected",
        "color": "green",
        "is_video": True,
        "snapshot_files": [],
        "original_filename": "Live Camera Stream",
        "identified_person": "Unknown",
        "match_image_file": None,
        "criminal_record_found": False
    }
    return render_template('live.html')

@app.route('/stop_live')
def stop_live():
    """Route that stops the live session and redirects to results."""
    global live_session_data
    session["result"] = live_session_data
    return redirect(url_for('results'))



if __name__ == "__main__":
    # use_reloader=False prevents the library-triggered reload loop (VGGFace.py)
    app.run(debug=True, use_reloader=False)