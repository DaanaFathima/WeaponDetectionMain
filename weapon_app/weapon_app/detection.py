"""
detection.py — Suspect face identification using OpenCV LBPH Face Recognizer.

No additional packages needed beyond opencv-python (already installed).

How it works:
  1. Load all images from the  suspect/  folder.
  2. Detect faces in each suspect image using Haar cascade.
  3. Train an LBPH recognizer on those faces.
  4. For any uploaded image/video, detect faces and predict identity.

Naming convention:
    suspect/john_doe.jpg   →  "John Doe"
    suspect/Jane Smith.png →  "Jane Smith"

Red bounding box  = matched suspect
Green bounding box = unknown person
"""

import os
import cv2
import numpy as np

SUSPECT_FOLDER  = "suspect"
CONFIDENCE_THRESHOLD = 80   # lower = stricter. LBPH returns 0 (perfect) → higher is worse


# ─────────────────────────────────────────────
#  Cascade loader (cached module-level)
# ─────────────────────────────────────────────
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def _detect_face_rects(gray_img):
    """Return list of (x, y, w, h) rectangles for detected faces."""
    faces = _face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    return faces if len(faces) > 0 else []


def _prepare_face(img_bgr, x, y, w, h, size=(160, 160)):
    """Crop, resize, equalise histogram for better recognition."""
    face = img_bgr[y : y + h, x : x + w]
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, size)
    face_eq = cv2.equalizeHist(face_resized)
    return face_eq


# ─────────────────────────────────────────────
#  Train recognizer from suspect folder
# ─────────────────────────────────────────────

def _build_recognizer():
    """
    Scan suspect/ folder, extract faces, train LBPH recognizer.
    Returns (recognizer, label_map {int → str}).
    """
    os.makedirs(SUSPECT_FOLDER, exist_ok=True)

    faces_data  = []
    labels_data = []
    label_map   = {}   # int → name
    label_id    = 0

    for fname in os.listdir(SUSPECT_FOLDER):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        name      = os.path.splitext(fname)[0].replace("_", " ").title()
        img_path  = os.path.join(SUSPECT_FOLDER, fname)
        img_bgr   = cv2.imread(img_path)

        if img_bgr is None:
            print(f"[detection] Could not read {fname}, skipping.")
            continue

        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        rects = _detect_face_rects(gray)

        if len(rects) == 0:
            # Try using the entire image as the face (useful for close-up portraits)
            h_img, w_img = gray.shape
            rects = [(0, 0, w_img, h_img)]
            print(f"[detection] No face detected in {fname}; using full image.")

        for (x, y, w, h) in rects:
            prepared = _prepare_face(img_bgr, x, y, w, h)
            faces_data.append(prepared)
            labels_data.append(label_id)

        label_map[label_id] = name
        label_id += 1
        print(f"[detection] Loaded suspect: {name}")

    if not faces_data:
        print("[detection] No suspect faces found. Identification will return 'Unknown'.")
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces_data, np.array(labels_data))
    print(f"[detection] Recognizer trained on {len(label_map)} suspect(s).")
    return recognizer, label_map


# ─────────────────────────────────────────────
#  Drawing helper
# ─────────────────────────────────────────────

def _draw_label(frame, x, y, w, h, name):
    color = (0, 0, 220) if name != "Unknown" else (0, 180, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label_y2 = y + h + 28
    # Clamp label so it stays inside frame
    if label_y2 > frame.shape[0]:
        label_y2 = y
        lbl_y1   = y - 28
    else:
        lbl_y1 = y + h
    cv2.rectangle(frame, (x, lbl_y1), (x + w, label_y2), color, cv2.FILLED)
    cv2.putText(frame, name, (x + 5, label_y2 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def _predict(recognizer, label_map, face_gray):
    """Run recognizer and return name string."""
    if recognizer is None:
        return "Unknown"
    try:
        label, confidence = recognizer.predict(face_gray)
        print(f"[detection] Predict: label={label}, confidence={confidence:.1f}")
        if confidence <= CONFIDENCE_THRESHOLD:
            return label_map.get(label, "Unknown")
    except Exception as e:
        print(f"[detection] Predict error: {e}")
    return "Unknown"


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def identify_faces_in_image(image_path):
    """
    Detect and identify suspects in a single image.

    Returns:
        annotated_image  (BGR numpy array)  — image with boxes and name labels
        identified_names (list[str])        — unique matched suspect names
    """
    recognizer, label_map = _build_recognizer()
    frame = cv2.imread(image_path)
    if frame is None:
        return None, []

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = _detect_face_rects(gray)
    identified = []

    for (x, y, w, h) in rects:
        face_eq = _prepare_face(frame, x, y, w, h)
        name    = _predict(recognizer, label_map, face_eq)
        if name != "Unknown":
            identified.append(name)
        _draw_label(frame, x, y, w, h, name)

    return frame, list(set(identified))


def identify_faces_in_video(video_path, output_path, process_every_n_frames=5):
    """
    Detect and identify suspects in a video.
    Saves annotated video to output_path.

    Returns:
        identified_names (list[str]) — unique suspect names across all frames
    """
    recognizer, label_map = _build_recognizer()

    cap    = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_identified = set()
    cached_results = []
    frame_idx      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % process_every_n_frames == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = _detect_face_rects(gray)
            cached_results = []

            for (x, y, w, h) in rects:
                face_eq = _prepare_face(frame, x, y, w, h)
                name    = _predict(recognizer, label_map, face_eq)
                cached_results.append((x, y, w, h, name))
                if name != "Unknown":
                    all_identified.add(name)

        for (x, y, w, h, name) in cached_results:
            _draw_label(frame, x, y, w, h, name)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return list(all_identified)


# ─────────────────────────────────────────────
#  Run standalone:  python detection.py <image_or_video_path>
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from datetime import datetime

    IDENTIFIED_FOLDER = "identified_suspect"
    os.makedirs(IDENTIFIED_FOLDER, exist_ok=True)

    def _make_save_path(names, ext):
        """Build a save path directly inside identified_suspect/."""
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        names_str = "_".join(names)
        filename  = f"{ts}_{names_str}{ext}"
        return os.path.join(IDENTIFIED_FOLDER, filename)

    if len(sys.argv) < 2:
        print("Usage:  python detection.py <path_to_image_or_video>")
        print("Example: python detection.py test.jpg")
        sys.exit(1)

    input_path = sys.argv[1]
    ext        = os.path.splitext(input_path)[1].lower()

    if ext in (".jpg", ".jpeg", ".png"):
        print(f"[main] Processing image: {input_path}")
        annotated, names = identify_faces_in_image(input_path)
        if names and annotated is not None:
            save_path = _make_save_path(names, ext)
            cv2.imwrite(save_path, annotated)
            print(f"[main] WARNING - Suspects identified: {', '.join(names)}")
            print(f"[main] SAVED -> {save_path}")
        else:
            print("[main] SAFE - No known suspects found. Nothing saved.")

    elif ext in (".mp4", ".avi", ".mov"):
        print(f"[main] Processing video: {input_path}")
        # For video we need a temp output path first, then rename into identified_suspect/
        tmp_path = os.path.join(IDENTIFIED_FOLDER, "_tmp_video" + ext)
        names = identify_faces_in_video(input_path, tmp_path)
        if names:
            save_path = _make_save_path(names, ext)
            os.rename(tmp_path, save_path)
            print(f"[main] WARNING - Suspects identified: {', '.join(names)}")
            print(f"[main] SAVED -> {save_path}")
        else:
            # Remove temp if no match
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            print("[main] SAFE - No known suspects found. Nothing saved.")

    else:
        print(f"[main] Unsupported file type: {ext}")
        sys.exit(1)


