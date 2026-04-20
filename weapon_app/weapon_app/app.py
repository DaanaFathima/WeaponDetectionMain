from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import os
import uuid
import cv2
import shutil
from datetime import datetime
from detection import identify_faces_in_image, identify_faces_in_video, _build_recognizer, _detect_face_rects, _prepare_face, _predict, _draw_label

app = Flask(__name__)

# Load trained YOLO model
model = YOLO("best.pt")

UPLOAD_FOLDER      = "uploads"
STATIC_FOLDER      = "static"
IDENTIFIED_FOLDER  = "identified_suspect"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(IDENTIFIED_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    threat_label = None
    alert_color = "green"
    result_file = None
    is_video = False
    snapshot_files = []
    identified_suspects = []     # NEW: list of identified suspect names

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            ext = file.filename.split(".")[-1].lower()
            filename = str(uuid.uuid4()) + "." + ext
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            # ================= IMAGE PROCESSING =================
            if ext in ["jpg", "jpeg", "png"]:
                # --- Weapon detection ---
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

                # --- Face / suspect identification ---
                try:
                    face_annotated, identified_suspects = identify_faces_in_image(upload_path)
                    # Overwrite the result image with face annotations layered on top
                    face_result_name = "face_" + filename
                    face_result_path = os.path.join(STATIC_FOLDER, face_result_name)
                    # Merge: draw weapon boxes on the face-annotated image
                    weapon_img = result_img.copy()
                    # Blend face annotations onto weapon-annotated image
                    import numpy as np
                    merged = cv2.addWeighted(weapon_img, 0.6, face_annotated, 0.4, 0)
                    cv2.imwrite(face_result_path, merged)
                    result_file = face_result_name
                except Exception as e:
                    print(f"[app] Face identification error: {e}")

                # --- Save to identified_suspect/ if match found ---
                if identified_suspects:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    names_str = "_".join(identified_suspects)
                    dest_name = f"{ts}_{names_str}_{result_file}"
                    shutil.copy(
                        os.path.join(STATIC_FOLDER, result_file),
                        os.path.join(IDENTIFIED_FOLDER, dest_name)
                    )
                    print(f"[app] Saved to identified_suspect/{dest_name}")

            # ================= VIDEO PROCESSING =================
            elif ext in ["mp4", "avi", "mov"]:
                is_video = True

                cap = cv2.VideoCapture(upload_path)
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps    = cap.get(cv2.CAP_PROP_FPS)

                output_name = str(uuid.uuid4()) + ".mp4"
                output_path = os.path.join(STATIC_FOLDER, output_name)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                detected_ids_total = set()
                frame_count = 0
                last_snapshot_frame = -9999

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1

                    results = model(frame, conf=0.25)
                    detected_ids = [int(box.cls[0]) for box in results[0].boxes]

                    # Detect gun (0) or knife (1). Save snapshot spaced out by at least 1 second (fps frames)
                    weapon_detected = 0 in detected_ids or 1 in detected_ids
                    if weapon_detected:
                        if frame_count - last_snapshot_frame >= fps:
                            snapshot_name = "snapshot_" + str(uuid.uuid4()) + ".jpg"
                            snapshot_path = os.path.join(STATIC_FOLDER, snapshot_name)
                            cv2.imwrite(snapshot_path, results[0].plot())
                            snapshot_files.append(snapshot_name)
                            last_snapshot_frame = frame_count

                    for cid in detected_ids:
                        detected_ids_total.add(cid)

                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)

                cap.release()
                out.release()

                # --- Weapon label ---
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

                # --- Face / suspect identification on original video ---
                try:
                    face_output_name = "face_" + output_name
                    face_output_path = os.path.join(STATIC_FOLDER, face_output_name)
                    identified_suspects = identify_faces_in_video(upload_path, face_output_path)
                    result_file = face_output_name
                except Exception as e:
                    print(f"[app] Face identification error: {e}")

                # --- Save to identified_suspect/ if match found ---
                if identified_suspects:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    names_str = "_".join(identified_suspects)
                    dest_name = f"{ts}_{names_str}_{result_file}"
                    shutil.copy(
                        os.path.join(STATIC_FOLDER, result_file),
                        os.path.join(IDENTIFIED_FOLDER, dest_name)
                    )
                    print(f"[app] Saved to identified_suspect/{dest_name}")

    return render_template(
        "index.html",
        result_file=result_file,
        threat=threat_label,
        color=alert_color,
        is_video=is_video,
        snapshot_files=snapshot_files,
        identified_suspects=identified_suspects,
    )

def gen_frames():
    """Generator function for live camera feed with weapon & suspect detection."""
    recognizer, label_map = _build_recognizer()
    cap = cv2.VideoCapture(0)
    frame_idx = 0
    cached_faces = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1. Weapon Detection
        results = model(frame, conf=0.25)
        # Using results[0].plot() to automatically draw the bounding boxes for weapons
        annotated_frame = results[0].plot()

        # 2. Suspect Identification (run every 5 frames for performance)
        if frame_idx % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = _detect_face_rects(gray)
            cached_faces = []
            for (x, y, w, h) in rects:
                face_eq = _prepare_face(frame, x, y, w, h)
                name = _predict(recognizer, label_map, face_eq)
                cached_faces.append((x, y, w, h, name))

        # Draw cached face bounding boxes on the annotated frame
        for (x, y, w, h, name) in cached_faces:
            _draw_label(annotated_frame, x, y, w, h, name)

        frame_idx += 1

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
    return render_template('live.html')


if __name__ == "__main__":
    app.run(debug=True)