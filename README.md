# Smart Attention & Emotion Tracker (Edge AI)

An advanced, privacy-preserving, Real-Time Edge AI system designed to monitor student engagement in remote educational environments. This system uses a lightweight PyTorch CNN (Multi-Task EfficientNet-V2) to run deep learning inference directly on the local machine. 

It provides an ultra-low latency WebSocket pipeline that routes live analytics and video feeds securely between Students, Teachers, and System Admins.

---

## 🏗️ Architecture & Technology Stack

The project has been completely overhauled from a basic monolithic script into a highly scalable **3-Tier FastAPI Architecture**:

### 1. Backend API & WebSockets (Python / FastAPI)
- **FastAPI:** High-performance async server handling HTTP routes and WebSocket connections.
- **Uvicorn:** Lightning-fast ASGI server for production-grade deployment.
- **OpenPyXL:** Pure-Python Excel parser (bypassing strict OS DLL blocks) used for automated Admin account generation.

### 2. Edge AI Inference (PyTorch)
- **PyTorch & Torchvision:** Powers the `MultiHeadEfficientNet` for Multi-Task Learning.
- **OpenCV (cv2):** Decodes raw Base64 video frames streamed from the student's browser.
- **Normalized Outputs:** The raw neural network emotion classes are mathematically scaled to an intuitive **0-100%** metric for all dashboards.
- **Privacy-First:** The AI runs locally. Video frames are analyzed instantly and discarded; only lightweight text-based telemetry is stored.

### 3. Database Layer (SQLite -> Ready for PyMongo)
- **SQLite:** Currently used for localized storage.
- **Abstracted Design:** The `database.py` layer is designed to return standard Python dictionaries. This means you can drop in PyMongo or PostgreSQL in the future without changing a single line of the main server logic!
- **Data Compression Algorithm:** The server automatically computes 1-Minute Rolling Averages of student emotions before writing to the database, saving 98% of potential storage space!

### 4. Frontend Portals (HTML5, JS, Chart.js)
Vanilla, zero-dependency HTML/JS/CSS served directly by FastAPI.
- **Admin Hub:** Features a massive `Chart.js` canvas for plotting live continuous class averages. It also includes an advanced SQL reporting engine that generates tables for Daily, Weekly, and Monthly historical analytics.
- **Teacher Dashboard:** Real-time, dynamic CSS Grid that automatically spawns a dedicated "Analytics Card" for every single student that joins the class. Clicking a specific student opens a low-latency WebSockets Video Modal so the teacher can monitor them privately.
- **Student Portal:** A strict waiting lobby that automatically hooks into the webcam ONLY when the Teacher globally clicks "Start Session".

---

## 🚀 Installation & Setup

1. **Clone the Repository** and open the folder in your terminal.
2. **Activate the Virtual Environment**:
   ```powershell
   .\.venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```powershell
   pip install fastapi uvicorn websockets openpyxl opencv-python torch torchvision
   ```
   *(Note: pandas has been explicitly removed from requirements to bypass Windows AppLocker restrictions).*

---

## 💻 Running the Server

Because this uses an ASGI asynchronous architecture, you **do not** run the python files directly. Start the server using Uvicorn:

```powershell
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🧪 Usage & Testing Guide

Once the server is running, navigate to `http://localhost:8000`. We have injected 3 default accounts so you can test the system immediately:

### 1. The Admin Portal (`admin` / `admin`)
- **Excel Registration:** Upload an Excel file containing columns for `Name`, `PRN`, and `Email`. The server will instantly generate secure passwords and download the results as a new Excel file.
- **Historical Reports:** Select Daily, Weekly, or Monthly to see the aggregated history of all student sessions!
- **Live Class Graph:** Watch the Chart.js line graph continuously plot class averages while a session is active. It maps the neural network's classifications to a beautiful 0-100% scale.

### 2. The Teacher Portal (`teacher` / `teacher`)
- Click **Start Live Session** to wake up the Student portals.
- The dynamic Grid will instantly spawn a tracking card for every student in the session.
- **Live Video Modal:** Click on any student's name card to instantly intercept their live webcam stream!
- **Automated Sleeping Alerts:** If a student closes their eyes for more than 120 continuous seconds, a massive red alert banner will flash on your screen.

### 3. The Student Portal (`student1` / `student`)
- Log in to be placed in the "Waiting Lobby".
- Once the Teacher starts the session, the browser will request webcam access.
- Analytics are transmitted at 1 frame-per-second to conserve massive amounts of bandwidth.
