# pose_extractor.py
import os
import cv2
import mediapipe as mp
import pickle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def extract_pose_from_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        pose_data = [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            }
            for lm in results.pose_landmarks.landmark
        ]
        return pose_data
    return None

def extract_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            frame_landmarks = [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }
                for lm in results.pose_landmarks.landmark
            ]
            all_landmarks.append(frame_landmarks)
    cap.release()

    if all_landmarks:
        num_landmarks = len(all_landmarks[0])
        avg_pose = []
        for i in range(num_landmarks):
            avg_pose.append({
                "x": sum([frame[i]["x"] for frame in all_landmarks]) / len(all_landmarks),
                "y": sum([frame[i]["y"] for frame in all_landmarks]) / len(all_landmarks),
                "z": sum([frame[i]["z"] for frame in all_landmarks]) / len(all_landmarks),
                "visibility": sum([frame[i]["visibility"] for frame in all_landmarks]) / len(all_landmarks),
            })
        return avg_pose
    return None

def save_pose_model(pose_landmarks, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(pose_landmarks, f)

def process_data_folder(data_folder, output_folder):
    for subdir, _, files in os.walk(data_folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file.endswith(('.jpg', '.png', '.jpeg', 'webp')):
                pose_landmarks = extract_pose_from_image(file_path)
            elif file.endswith('.mp4'):
                pose_landmarks = extract_pose_from_video(file_path)
            else:
                continue
            if pose_landmarks:
                output_path = os.path.join(output_folder, f"{os.path.basename(file)}.pkl")
                save_pose_model(pose_landmarks, output_path)
                print(f"Processed: {file}")

if __name__ == "__main__":
    process_data_folder('data', 'models')
