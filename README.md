# Real-Time Multi-Task Attention & Emotion Tracker

A sophisticated self-monitoring and classroom analytics tool powered by **PyTorch** and **EfficientNet-V2-S**. This application uses a webcam to perform real-time, multi-task facial analysis. 

It simultaneously tracks:
- **Eye State** (Sleeping / Awake)
- **Boredom**
- **Engagement** (Interesting)
- **Frustration**
- **Confusion**

The system features a beautiful, responsive dark-mode dashboard (with premium glassmorphism UI) for both students and teachers, and includes automated alerts (e.g., triggering a "SLEEPING DETECTED" alert if eyes are closed for more than 2 minutes).

---

## 🚀 Features

- **True Multi-Task Learning**: The neural network uses a single EfficientNet-V2-S backbone that splits into 5 separate classification heads, allowing it to predict eye state and 4 distinct emotions simultaneously in real-time.
- **Robust Eye Tracking**: Trained with heavy data augmentations (RandomCrop, ColorJitter, Rotation) to accurately detect closed eyes even if the user's face is slightly misaligned or tilted.
- **Automated Data Pipeline**: Seamlessly merges and balances the DAiSEE (video) and OACE (image) datasets to ensure zero class bias.
- **Performance Plotting**: Automatically generates `matplotlib` accuracy and loss graphs after training.
- **Web Dashboards**: Includes both a Student Portal (live tracking) and an Admin Portal (monitoring multiple students).

---

## 🛠️ Setup & Installation

### 1. Environment Configuration
This project is built on **Python 3.11** (Windows). 
Create a virtual environment and install the required dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install matplotlib
```

*(Note: Ensure your `.venv` is strictly named `.venv` to align with the `.gitignore` rules).*

---

## 📊 Dataset Preparation

The AI is trained on a combination of the **DAiSEE** dataset (for complex classroom emotions) and the **OACE** dataset (for highly accurate open/closed eye states).

Ensure your data is located here (or update the paths in the script):
- OACE Images: `C:\Users\HP\OneDrive\Desktop\archive (1)\OACE`
- DAiSEE Videos: `D:\AI\Project\Dataset\DAiSEE\DAiSEE\DataSet`

### Build the Dataset
Run the data preparation script to extract frames from up to 1000 DAiSEE videos and perfectly balance them with the OACE images:

```powershell
.\.venv\Scripts\python src\prepare_custom_data.py
```
*This will automatically split the data into 70% Training, 15% Validation, and 15% Testing.*

---

## 🧠 Training (Fine-Tuning)

Instead of training from scratch, the model utilizes **True Fine-Tuning**. The base feature extraction layers of the EfficientNet-V2 are frozen, and only the final few blocks and custom emotion heads are trained. This prevents overfitting and trains significantly faster.

To train the model (utilizing `num_workers=4` for fast data loading and a batch size of 128 to maximize GPU VRAM):

```powershell
.\.venv\Scripts\python src\train.py --batch-size 128 --learning-rate 0.0005 --epochs 30
```

When training completes (or triggers early stopping), the model will be saved to `models/attention_cnn.pt` and a visual performance graph will be saved to `models/training_results.png`.

---

## 🌐 Running the Web Application

Start the local web server to access the dashboards:

```powershell
.\.venv\Scripts\python src\web_app.py
```

- **User Portal**: [http://127.0.0.1:8010](http://127.0.0.1:8010)
- **Admin Portal**: [http://127.0.0.1:8020](http://127.0.0.1:8020)

### UI Metrics
The dashboard will display real-time readouts for:
- **Boring** (None / Low / High / Max)
- **Interesting** (None / Low / High / Max)
- **Frustrated** (None / Low / High / Max)
- **Sleeping** (Yes / No)
- **No Face** (Yes / No)

*Press `Ctrl+C` in the terminal to stop the server.*
