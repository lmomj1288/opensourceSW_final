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
        self.match_duration = 0
        self.blend_alpha = 0.0  # 블렌딩 강도를 위한 변수

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

        landmarks1_norm = landmarks1 - np.mean(landmarks1, axis=0)
        landmarks2_norm = landmarks2 - np.mean(landmarks2, axis=0)

        scale1 = np.sqrt(np.sum(landmarks1_norm ** 2))
        scale2 = np.sqrt(np.sum(landmarks2_norm ** 2))
        
        landmarks1_scaled = landmarks1_norm / scale1
        landmarks2_scaled = landmarks2_norm / scale2

        diff = landmarks1_scaled - landmarks2_scaled
        distance = np.sqrt(np.sum(diff ** 2))
        
        similarity = 1 / (1 + distance)
        return similarity

    def create_body_mask(self, landmarks, image_shape):
        """랜드마크를 기반으로 신체 마스크 생성"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * image_shape[1])
            y = int(landmark.y * image_shape[0])
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.fillPoly(mask, [hull], 255)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask / 255.0

    def create_template(self, image_path):
        """템플릿 이미지의 pose landmarks 추출"""
        template_image = cv2.imread(image_path)
        if template_image is None:
            raise ValueError("템플릿 이미지를 불러올 수 없습니다.")
            
        template_image = cv2.resize(template_image, (640, 480))
        self.template_landmarks, template_pose_landmarks = self.extract_pose_landmarks(template_image)
        self.template_image = template_image
        
        if self.template_landmarks is None:
            raise ValueError("템플릿 이미지에서 포즈를 감지할 수 없습니다.")

    def apply_blending_effect(self, frame, template_image, similarity, pose_landmarks):
        """블렌딩 효과 적용"""
        if similarity >= 0.80:  # 임계값을 0.80(80%)로 설정
            self.match_duration += 1
            # 블렌딩 강도를 부드럽게 증가
            self.blend_alpha = min(0.8, self.blend_alpha + 0.05)
        else:
            self.match_duration = 0
            # 블렌딩 강도를 부드럽게 감소
            self.blend_alpha = max(0.0, self.blend_alpha - 0.05)

        if self.blend_alpha > 0 and pose_landmarks:
            # 신체 마스크 생성
            body_mask = self.create_body_mask(pose_landmarks, frame.shape)
            mask3d = np.stack([body_mask] * 3, axis=2)
            
            # 블렌딩 적용
            blended = template_image.copy()
            np.copyto(blended, frame, where=(mask3d > 0.5))
            
            # 최종 이미지 생성
            output = cv2.addWeighted(
                template_image, 1 - self.blend_alpha,
                blended, self.blend_alpha,
                0
            )

            if self.match_duration > 30:
                # 성공 효과
                h, w = frame.shape[:2]
                glow = np.zeros_like(frame, dtype=np.float32)
                cv2.circle(glow, (w//2, h//2), 
                          int(min(w,h) * 0.4), 
                          (0.5, 1.0, 0.5), -1)
                glow = cv2.GaussianBlur(glow, (21, 21), 0)
                output = cv2.addWeighted(output, 1.0, glow, 0.3, 0)

            return output
        
        return template_image.copy()

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

                current_landmarks, current_pose_landmarks = self.extract_pose_landmarks(frame)
                output_image = np.zeros((480, 1280, 3), dtype=np.uint8)

                if current_landmarks is not None and current_pose_landmarks is not None:
                    similarity = self.calculate_pose_similarity(
                        self.template_landmarks, 
                        current_landmarks
                    )

                    # 블렌딩 효과 적용
                    blended_template = self.apply_blending_effect(
                        frame, self.template_image, similarity, current_pose_landmarks
                    )
                    output_image[:, :640] = blended_template
                    
                else:
                    output_image[:, :640] = self.template_image

                # 웹캠 영상 표시
                output_image[:, 640:] = frame

                # 포즈 랜드마크 시각화
                if current_pose_landmarks:
                    self.mp_draw.draw_landmarks(
                        output_image[:, 640:],
                        current_pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    )

                # 유사도 점수 표시
                if current_landmarks is not None:
                    score_text = f"Similarity: {similarity:.2%}"
                    cv2.putText(output_image, score_text,
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (0, 255, 0), 2)

                    if similarity >= 0.80:
                        cv2.putText(output_image, "MATCH!",
                                  (640 + 200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                  1.5, (0, 255, 0), 3)

                # 설명 텍스트 추가
                cv2.putText(output_image, "Template Pose", (10, 450), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_image, "Your Pose", (650, 450), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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