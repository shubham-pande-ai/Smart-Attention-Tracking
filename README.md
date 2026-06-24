# Real-Time Eye Gaze Attention Tracker

Self-monitoring tool for online classes. It classifies cropped eye images into:

- `looking_screen`
- `looking_away`
- `eyes_closed`

The GUI uses the webcam, predicts the current attention state, and keeps a live session attention score.
If the student keeps looking away or appears sleepy for a sustained period, the app can notify the teacher.

## Kaggle Datasets

Recommended datasets:

1. MPII Gaze / MPIIFaceGaze for gaze direction:
   https://www.kaggle.com/datasets/tntg/mpii-gaze

   MPIIGaze is a real-world appearance-based gaze estimation dataset and is large enough for CNN or transfer-learning experiments. The official dataset page describes it as real-world gaze data collected with monocular RGB cameras.

2. MRL Eye Dataset for open/closed eyes:
   https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset

   This Kaggle fork lists more than 85,000 eye images split into awake/sleepy classes, which is useful for the `eyes_closed` part of this project.

Optional gaze-direction alternative:

- GazeTrack:
  https://www.kaggle.com/datasets/tooyoungalex/gazetrack

  Kaggle lists about 473k single-eye files from 47 volunteers, so it is also large enough for training. This is a large download, around 8 GB as a zip file.

## Project Structure

```text
data/
  raw/                 # Kaggle downloads/unzipped datasets
  processed/
    train/
      looking_screen/
      looking_away/
      eyes_closed/
    val/
      looking_screen/
      looking_away/
      eyes_closed/
    test/
      looking_screen/
      looking_away/
      eyes_closed/
models/
  attention_cnn.keras
src/
  download_datasets.py
  product_app.py
  web_app.py
  prepare_gazetrack_sources.py
  train.py
  prepare_imagefolder.py
```

## Setup

Use Python 3.11 on Windows. TensorFlow may not install on newer Python versions such as 3.13 or 3.14.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Download Data From Kaggle

Install and configure the Kaggle API first:

```powershell
pip install kaggle
mkdir $env:USERPROFILE\.kaggle
# Put kaggle.json from your Kaggle account settings into %USERPROFILE%\.kaggle\
```

Then download:

```powershell
kaggle datasets download -d tntg/mpii-gaze -p data/raw/mpii-gaze --unzip
kaggle datasets download -d tooyoungalex/gazetrack -p data/raw/gazetrack --unzip
kaggle datasets download -d akashshingha850/mrl-eye-dataset -p data/raw/mrl-eye-dataset --unzip
```

Or use the project helper:

```powershell
python src/download_datasets.py mrl gazetrack
```

If a dataset returns `403 Forbidden`, open the Kaggle dataset page in your browser, sign in, accept any dataset terms, and rerun the command.

## Prepare Training Folders

Because Kaggle gaze datasets vary in folder/annotation format, the included prep script supports the final supervised image-folder layout. Put or export images into three source folders, then split them:

```powershell
python src/prepare_imagefolder.py `
  --looking-screen data/source/looking_screen `
  --looking-away data/source/looking_away `
  --eyes-closed data/source/eyes_closed `
  --output data/processed
```

For a strong student-project baseline:

- Use MPII/GazeTrack eye crops whose gaze target is near the screen/camera center as `looking_screen`.
- Use MPII/GazeTrack eye crops with large horizontal/vertical gaze offsets as `looking_away`.
- Use MRL `Sleepy` or closed-eye images as `eyes_closed`.

If MPII/GazeTrack is unavailable, do not fake `looking_screen` and `looking_away` from the same awake-eye images for final accuracy. That makes the two classes visually almost identical and accuracy will usually stop near 65-67%.

If GazeTrack has been downloaded, convert its coordinate-labeled eye frames into source folders:

```powershell
python src/prepare_gazetrack_sources.py `
  --gazetrack-root data/raw/gazetrack/GazeTrack `
  --output data/source `
  --max-per-class 6000 `
  --clean-gaze
```

Then rebuild the processed train/validation/test folders:

```powershell
python src/prepare_imagefolder.py `
  --looking-screen data/source/looking_screen `
  --looking-away data/source/looking_away `
  --eyes-closed data/source/eyes_closed `
  --output data/processed `
  --clean
```

## Collect Personal Calibration Data

For a reliable demo, collect webcam eye crops with real labels:

```powershell
python src/collect_calibration_data.py --class-name looking_screen --count 600
python src/collect_calibration_data.py --class-name looking_away --count 600
python src/collect_calibration_data.py --class-name eyes_closed --count 600
```

During collection:

- for `looking_screen`, look directly at the online-class screen
- for `looking_away`, look left/right/down or away from the screen
- for `eyes_closed`, close your eyes or act sleepy
- press `S` to start/pause recording and `Q` to quit

Then rebuild the processed dataset:

```powershell
python src/prepare_imagefolder.py `
  --looking-screen data/source/looking_screen `
  --looking-away data/source/looking_away `
  --eyes-closed data/source/eyes_closed `
  --output data/processed `
  --clean
```

## Train

```powershell
python src/train.py --data data/processed --model-out models/attention_cnn.keras --epochs 15
```

The model uses EfficientNetV2B0 transfer learning by default for better accuracy:

```powershell
python src/train.py --data data/processed --model-out models/attention_cnn.keras --epochs 20 --architecture efficientnetv2b0
```

For older or slower machines, use MobileNetV2:

```powershell
python src/train.py --data data/processed --model-out models/attention_cnn.keras --epochs 15 --architecture mobilenetv2
```

## Run Web Application

The application runs as a local web service consisting of two web portals: Student (User) and Admin (Teacher/Main Admin).

To start the web portals, run:

```powershell
python src/web_app.py
```

*Note: Since the desktop GUI has been removed, running `python src/product_app.py` will also automatically start the web portals.*

For presentation in a browser, run the local web portals:

```powershell
python src/web_app.py
```

This starts two localhost portals from one process:

- User portal: http://127.0.0.1:8010
- Admin portal: http://127.0.0.1:8020

Use the User portal for each student browser/tab. Use the Admin portal for classroom monitoring (Teacher), and administrative control (Main Admin).

Default local demo accounts:

- User: `user` / `user123`
- Admin: `admin` / `admin123`
- Main Admin: `mainadmin` / `mainadmin123` (logs in via the Admin portal)

User mode:

- sign in
- give webcam consent before starting
- start/end an attention session
- see live attention score, confidence, current state, no-face/uncertain state, and soft alert status
- save session history automatically
- delete personal session history

Admin/Teacher mode:

- view live user sessions during the lecture
- see in-app teacher alerts
- acknowledge alerts
- view recent session history

Main Admin mode (also accessed via the Admin Portal):

- add/deactivate User and Admin accounts
- create classes
- assign users to classes
- view class roster and latest attention status
- view session history
- view and acknowledge alert history
- export session reports as CSV or PDF
- configure model path, confidence threshold, note-taking grace, eyes-closed threshold, alert cooldown, and email settings

SQLite database:

```text
data/attention_tracker.sqlite3
```

Tables created by the app:

- `users`
- `students`
- `teachers`
- `classes`
- `enrollments`
- `attention_sessions`
- `attention_events`
- `alerts`
- `alert_settings`
- `settings`

## Teacher Notification

By default, the web application handles student distraction and sleepy alerts via the SQLite database and displays them live on the **Admin Portal** dashboard.

The alert logic is tuned so normal note-taking does not immediately notify the teacher:

- **Looking Away**: Default grace period is `180` seconds (configured as `note_taking_grace` in settings), so looking down to write notes is tolerated.
- **Eyes Closed / Sleepy**: Default alert threshold is `20` seconds (configured as `eyes_closed_alert_after` in settings).
- **Cooldown**: Repeated alerts wait at least `300` seconds (configured as `alert_cooldown` in settings) before triggering again.

All of these thresholds, along with the SMTP mail settings (for optional email notifications sent to the teacher), can be configured dynamically in the **Admin Portal** settings tab.

Email alerts can be configured with the following settings on the Admin Portal:
- `alert_mode`: Set to `email`.
- `teacher_email`: Destination email address.
- `smtp_host`, `smtp_port`, `smtp_user`, `smtp_password`: Your SMTP email provider credentials.

Attention scoring:

- `looking_screen`: 1.0
- `looking_away`: 0.25
- `eyes_closed`: 0.0

The displayed score is the mean over the running webcam session.

## Product-Quality Model Roadmap

For a Zoom-like product, use a two-stage model instead of relying only on the current Haar-cascade eye crop and 3-class CNN:

1. Use MediaPipe Face Landmarker for robust face landmarks, eye landmarks, face tracking, and head pose signals.
2. Train a gaze-direction model on real gaze datasets such as GazeTrack or MPIIGaze.
3. Keep the current `eyes_closed` classifier or replace it with an Eye Aspect Ratio / blink model from landmarks.
4. Fuse the signals with time rules:
   - short downward gaze = possible note-taking
   - long downward/side gaze = soft alert
   - long eyes-closed = stronger sleepy alert
   - no face detected = separate session quality signal, not the same as looking away
5. Add per-student calibration so the model learns each student's normal webcam angle, glasses, lighting, and note-taking posture.

For future Zoom integration, keep webcam analysis local in the student app and send only session metrics or soft alerts to the teacher/admin dashboard.

## Privacy Note

This is designed as a self-monitoring productivity tool. Webcam frames stay local; the GUI does not upload or save video.
