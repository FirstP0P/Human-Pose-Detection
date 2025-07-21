import cv2
import mediapipe as mp
import json
from datetime import datetime
import os

class BodyLandmarkCapture:
    # Adjusted landmark indices for more comprehensive body detection
    LANDMARK_INDICES = {
        'Nose': 0,
        'Left_Eye': 1,
        'Right_Eye': 2,
        'Left_Ear': 3,
        'Right_Ear': 4,
        'Left_Shoulder': 5,
        'Right_Shoulder': 6,
        'Left_Elbow': 7,
        'Right_Elbow': 8,
        'Left_Wrist': 9,
        'Right_Wrist': 10,
        'Left_Hip': 11,
        'Right_Hip': 12,
        'Left_Knee': 13,
        'Right_Knee': 14,
        'Left_Ankle': 15,
        'Right_Ankle': 16,
        'Left_Heel': 23,
        'Right_Heel': 24,
        'Left_Toe': 21,
        'Right_Toe': 22,
        'Left_Foot_Index': 19,
        'Right_Foot_Index': 20,
    }

    def __init__(self):
        self.mp_pose = mp.solutions.pose  # type: ignore
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Use 2 for higher accuracy
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.body_data = []
        self.frame_count = 0

    def capture_landmarks(self):
        cap = cv2.VideoCapture(r"body_video.mp4")  # Change to 0 for webcam

        if not cap.isOpened():
            raise RuntimeError("Could not open video capture device")

        try:
            while True:
                success, image = cap.read()
                if not success:
                    print("Failed to capture frame")
                    break  # Break on failure (end of video)

                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)

                if results.pose_landmarks:
                    self.process_landmarks(image, results.pose_landmarks)

                cv2.imshow('Body Landmark Capture', image)

                if cv2.waitKey(5) & 0xFF == ord('w'):
                    break

                self.frame_count += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.save_body_data()

    def process_landmarks(self, image, pose_landmarks):
        height, width, _ = image.shape
        current_frame = {
            'frame': self.frame_count,
            'landmarks': {}
        }

        for feature, idx in self.LANDMARK_INDICES.items():
            landmark = pose_landmarks.landmark[idx]
            current_frame['landmarks'][feature] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'pixel_x': int(landmark.x * width),
                'pixel_y': int(landmark.y * height)
            }
            cv2.circle(image,
                       (int(landmark.x * width), int(landmark.y * height)),
                       2, (0, 255, 0), -1)

        self.body_data.append(current_frame)

    def save_body_data(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'output'

        # Create the folder if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f'body_data_{timestamp}.json')
        with open(filename, 'w') as f:
            json.dump({
                'metadata': {
                    'frame_count': self.frame_count,
                    'timestamp': timestamp,
                    'landmark_indices': self.LANDMARK_INDICES
                },
                'frames': self.body_data
            }, f, indent=2)

        print(f"✅ Body data saved to {filename}")


if __name__ == "__main__":
    capturer = BodyLandmarkCapture()
    capturer.capture_landmarks()
