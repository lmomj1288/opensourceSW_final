# pose_extractor.py
import os
import cv2
import mediapipe as mp
import pickle

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def extract_pose_from_image(image_path):
    """이미지에서 포즈를 추출"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        # 포즈 데이터를 리스트로 변환
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
    """동영상에서 평균 포즈를 추출"""
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            # 프레임별 포즈 데이터를 리스트로 변환
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
        # 모든 프레임의 평균 포즈 계산
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
    """포즈 데이터를 모델로 저장"""
    with open(output_path, 'wb') as f:
        pickle.dump(pose_landmarks, f)

def process_data_folder(data_folder, output_folder):
    """폴더 내의 모든 데이터 처리"""
    for subdir, _, files in os.walk(data_folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file.endswith(('.jpg', '.png', '.jpeg')):
                pose_landmarks = extract_pose_from_image(file_path)
            elif file.endswith('.mp4'):
                pose_landmarks = extract_pose_from_video(file_path)
            else:
                continue
            if pose_landmarks:
                # 파일명 기반으로 출력 경로 설정
                output_path = os.path.join(output_folder, f"{os.path.basename(file)}.pkl")
                save_pose_model(pose_landmarks, output_path)
                print(f"Processed: {file}")

if __name__ == "__main__":
    # 데이터 폴더와 모델 폴더 경로
    process_data_folder('data', 'models')
