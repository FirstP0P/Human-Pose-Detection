Sure! Here's a complete and professional `README.md` tailored for your **Body Landmark Capture** project.

---

### 📄 `README.md`

````markdown
# Body Landmark Capture

This Python script uses [MediaPipe](https://github.com/google/mediapipe) and [OpenCV](https://opencv.org/) to capture body pose landmarks from a video file or webcam in real time. It saves the extracted landmark data to a JSON file for further analysis or machine learning use.

---

## 📌 Features

- Tracks 25+ body landmarks including head, shoulders, hips, knees, ankles, and feet
- Saves landmarks with both normalized (0-1) and pixel coordinates
- JSON output with metadata and frame-wise landmark breakdown
- Works with videos or live webcam feed
- Automatically creates `output/` folder for saved files

---

## 🖥️ Requirements

Install the dependencies using pip:

```bash
pip install opencv-python mediapipe
````

---

## 🚀 Usage

### 1. Place your input video

Place your input video file (e.g., `body_video.mp4`) in the project directory or update the path in the script.

### 2. Run the script

```bash
python body_landmark_capture.py
```

Press **`W`** while the video window is active to stop the capture and save the results.

---

## 📂 Output

The script saves a JSON file to the `output/` directory with the following structure:

```json
{
  "metadata": {
    "frame_count": 123,
    "timestamp": "20250721_143210",
    "landmark_indices": {
      "Nose": 0,
      "Left_Eye": 1,
      ...
    }
  },
  "frames": [
    {
      "frame": 0,
      "landmarks": {
        "Nose": {
          "x": 0.52,
          "y": 0.33,
          "z": -0.10,
          "pixel_x": 312,
          "pixel_y": 198
        },
        ...
      }
    },
    ...
  ]
}
```

---

## 🎯 Customization

* To use a **webcam**, change this line:

  ```python
  cap = cv2.VideoCapture("body_video.mp4")
  ```

  to:

  ```python
  cap = cv2.VideoCapture(0)
  ```

* To increase detection accuracy, you can set `model_complexity=2` in the `self.mp_pose.Pose(...)` call.

---

## 🛠️ Project Structure

```
body_landmark_capture.py    # Main script
output/                     # JSON data saved here
README.md                   # This file
```

---
