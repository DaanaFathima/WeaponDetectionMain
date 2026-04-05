# ThreatScan — AI-Powered Weapon & Suspect Identification System

ThreatScan is an advanced security application that performs real-time weapon detection and facial identification using artificial intelligence. 

## 🧠 Project Core Concepts
The application operates on two primary intelligence layers:
1. **Weapon Detection (YOLOv8):** Analyzes uploaded images and videos to identify firearms (Guns) and bladed weapons (Knives) with bounding box visualization.
2. **Facial Identification (DeepFace):** Automatically cross-references detected faces in the media against the `known_faces/` database (acts as a "criminal record" system).

## 🛠 Prerequisites

- **Python Version:** **Python 3.10** (Recommended)
- **Dependencies:** Listed in `requirements.txt`.

## 🚀 Getting Started

### 1. Create a Virtual Environment
It is highly recommended to use a virtual environment to avoid package conflicts:
```powershell
# Open terminal in the project root (weapon_app folder)
python -m venv Env
```

### 2. Activate the Environment
- **Windows:**
  ```powershell
  .\Env\Scripts\activate
  ```
- **Linux/macOS:**
  ```bash
  source Env/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```
The server will start at `http://127.0.0.1:5000`.

## 📂 Project Structure

- `app.py`: The main Flask server logic.
- `best.pt`: The trained YOLOv8 model weights for weapon detection.
- `known_faces/`: A folder containing images of individuals to be identified (e.g., `MONICA.jpg`). The file name determines the person's identity.
- `uploads/`: Temporary storage for uploaded media.
- `static/`: Stores processed results and detection snapshots.
- `templates/`: HTML front-end designs (Detection and Results pages).

## 🛡 How False Positives are Managed
The system includes high-level filters to prevent common errors (e.g., confusing humans with guns based on vertical aspect ratios). If a detection meets certain "human-like" dimensions, it is automatically stripped by the backend to ensure high accuracy.

## 👥 Criminal Records
To add someone to the identification system, simply drop a clear headshot image into the `known_faces/` folder. The system will automatically use the image filename as the person's name for matches.
