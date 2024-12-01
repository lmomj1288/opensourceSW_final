import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

class PoseMatchingSystem:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.template_landmarks = None

    def extract_pose_landmarks(self, image):
        """이미지에서 pose landmarks 추출"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks), results.pose_landmarks
        return None, None

    def calculate_pose_similarity(self, landmarks1, landmarks2):
        """두 포즈 간의 유사도 계산"""
        if landmarks1 is None or landmarks2 is None:
            return 0.0

        # 좌표 정규화
        landmarks1_norm = landmarks1 - np.mean(landmarks1, axis=0)
        landmarks2_norm = landmarks2 - np.mean(landmarks2, axis=0)

        # Procrustes 분석을 통한 유사도 계산
        scale1 = np.sqrt(np.sum(landmarks1_norm ** 2))
        scale2 = np.sqrt(np.sum(landmarks2_norm ** 2))
        
        landmarks1_scaled = landmarks1_norm / scale1
        landmarks2_scaled = landmarks2_norm / scale2

        # 유클리드 거리 계산
        diff = landmarks1_scaled - landmarks2_scaled
        distance = np.sqrt(np.sum(diff ** 2))
        
        # 거리를 0-1 사이의 유사도 점수로 변환
        similarity = 1 / (1 + distance)
        return similarity

    def create_template(self, image_path):
        """템플릿 이미지의 pose landmarks 추출"""
        template_image = cv2.imread(image_path)
        if template_image is None:
            raise ValueError("템플릿 이미지를 불러올 수 없습니다.")
            
        # 템플릿 이미지 크기 조정 (웹캠 크기에 맞춤)
        template_image = cv2.resize(template_image, (640, 480))
        self.template_landmarks, template_pose_landmarks = self.extract_pose_landmarks(template_image)
        self.template_image = template_image
        
        # 템플릿 이미지에 포즈 랜드마크 그리기
        template_with_pose = template_image.copy()
        if template_pose_landmarks:
            self.mp_draw.draw_landmarks(
                template_with_pose,
                template_pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        self.template_with_pose = template_with_pose
        
        if self.template_landmarks is None:
            raise ValueError("템플릿 이미지에서 포즈를 감지할 수 없습니다.")

    def run_webcam_matching(self):
        """웹캠을 통한 실시간 포즈 매칭"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("웹캠에서 프레임을 읽을 수 없습니다.")
                    break

                # 현재 프레임의 포즈 감지
                current_landmarks, current_pose_landmarks = self.extract_pose_landmarks(frame)
                
                # 결과 이미지 준비 (웹캠과 템플릿 이미지를 나란히 표시)
                output_image = np.zeros((480, 1280, 3), dtype=np.uint8)
                
                # 템플릿 이미지 왼쪽에 표시
                output_image[:, :640] = self.template_with_pose
                
                # 웹캠 이미지 오른쪽에 표시
                output_image[:, 640:] = frame
                
                if current_landmarks is not None and current_pose_landmarks is not None:
                    # 포즈 유사도 계산
                    similarity = self.calculate_pose_similarity(
                        self.template_landmarks, 
                        current_landmarks
                    )
                    
                    # 웹캠 영상에 포즈 시각화
                    self.mp_draw.draw_landmarks(
                        output_image[:, 640:],
                        current_pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
                    
                    # 유사도 점수 표시
                    score_text = f"Similarity: {similarity:.2%}"
                    cv2.putText(output_image, score_text,
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0, 255, 0), 2)
                    
                    # 95% 이상 매칭되면 1 출력
                    if similarity >= 0.95:
                        cv2.putText(output_image, "1",
                                  (640 + 320, 240), cv2.FONT_HERSHEY_SIMPLEX,
                                  4, (0, 255, 0), 4)

                # 종료 안내 텍스트
                cv2.putText(output_image, "Press 'q' to quit",
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (0, 255, 0), 2)

                # 구분선 그리기
                cv2.line(output_image, (640, 0), (640, 480), (255, 255, 255), 2)

                cv2.imshow('Pose Matching', output_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()

def main():
    image_path = "C:/Users/lmomj/Desktop/opensource/final/movies/bakha.jpg"
    
    try:
        system = PoseMatchingSystem()
        print("템플릿 포즈 추출 중...")
        system.create_template(image_path)
        
        print("웹캠 매칭 시작...")
        system.run_webcam_matching()

    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == '__main__':
    main()