import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

class PoseMatchingSystem:
    def __init__(self):
        # DeepLab-V3+ 모델 로드
        self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # MediaPipe Pose 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 이미지 전처리를 위한 변환 정의
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def check_arms_spread(self, landmarks):
        """팔을 벌린 자세인지 확인"""
        if landmarks is None:
            return False
        
        # 필요한 랜드마크 좌표 추출
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
    
        # 팔의 각도 계산
        def calculate_angle(a, b, c):
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])
            
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
                
            return angle

        # 수평 각도 계산 (어깨-손목)
        def calculate_horizontal_angle(shoulder, wrist):
            dx = wrist.x - shoulder.x
            dy = wrist.y - shoulder.y
            angle = np.abs(np.degrees(np.arctan2(dy, dx)))
            return angle

        # 팔꿈치가 펴져있는지 확인 (팔꿈치 각도)
        left_arm_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)
        right_arm_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)

        # 수평으로 펴져있는지 확인
        left_horizontal = calculate_horizontal_angle(left_shoulder, left_wrist)
        right_horizontal = calculate_horizontal_angle(right_shoulder, right_wrist)

        # 조건 검사
        # 1. 팔꿈치가 충분히 펴져있는지 (165도 이상)
        arms_straight = (left_arm_angle > 165 and right_arm_angle > 165)
        
        # 2. 팔이 수평에 가까운지 (수평 = 0도 또는 180도)
        left_is_horizontal = (left_horizontal < 20 or left_horizontal > 160)
        right_is_horizontal = (right_horizontal < 20 or right_horizontal > 160)
        arms_horizontal = left_is_horizontal and right_is_horizontal
        
        # 3. 양팔의 높이가 비슷한지
        height_difference = abs(left_wrist.y - right_wrist.y)
        similar_height = height_difference < 0.05  # 더 엄격한 높이 차이 제한
        
        # 4. 팔이 충분히 벌어져 있는지
        width_difference = abs(left_wrist.x - right_wrist.x)
        arms_spread_wide = width_difference > 0.4  # 최소 벌림 너비 설정
        
        # 모든 조건을 만족해야 성공
        return arms_straight and arms_horizontal and similar_height and arms_spread_wide

    def process_frame(self, frame, bg_color=(192, 192, 192)):
        # 포즈 감지
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(frame_rgb)
        
        # 포즈 매칭 점수 계산
        pose_score = 1 if (pose_results.pose_landmarks and 
                          self.check_arms_spread(pose_results.pose_landmarks)) else 0
        
        # 세그멘테이션 처리
        image = Image.fromarray(frame_rgb)
        width, height = 640, 480
        image = image.resize((width, height))
        
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        output_predictions = output.argmax(0).cpu().numpy()
        mask = np.zeros_like(output_predictions)
        mask[output_predictions == 15] = 1

        mask = cv2.medianBlur(mask.astype(np.uint8), 7)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (7,7), 0)

        frame_resized = cv2.resize(frame, (width, height))
        bg_image = np.zeros_like(frame_resized)
        bg_image[:] = bg_color
        mask_3channel = np.stack([mask] * 3, axis=2)
        
        output_image = cv2.convertScaleAbs(
            frame_resized * mask_3channel + bg_image * (1 - mask_3channel)
        )

        # 포즈 매칭 점수와 가이드라인 표시
        cv2.putText(output_image, f"Pose Score: {pose_score}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 포즈가 인식되면 추가 정보 표시
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            height_diff = abs(left_wrist.y - right_wrist.y)
            width_diff = abs(left_wrist.x - right_wrist.x)
            
            info_text = f"Height Diff: {height_diff:.3f}, Width: {width_diff:.3f}"
            cv2.putText(output_image, info_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return output_image, pose_score

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    system = PoseMatchingSystem()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break

            output_frame, pose_score = system.process_frame(frame)
            combined_frame = np.hstack((frame, output_frame))
            cv2.imshow('Pose Matching (Original | Segmented)', combined_frame)

            # 포즈 매칭 결과 출력
            if pose_score == 1:
                print("박하사탕 포즈 매칭 성공!")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"에러 발생: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()