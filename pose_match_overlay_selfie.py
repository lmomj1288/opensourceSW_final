import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import pyttsx3 
import time

class PoseMatchingSystem:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.selfie_seg = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.mp_draw = mp.solutions.drawing_utils
        self.current_index = 0
        self.template_images = []
        self.template_landmarks = []
        self.template_pose_landmarks = []
        self.match_duration = 0
        self.blend_alpha = 0.0
        self.similarity = 0.0
        self.success_start_time = None
        self.success_duration = 3.0    # 3초 유지
        self.complete_time = None
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 200)
        self.engine.setProperty('volume', 0.9)
        self.completion_announced = False 
        self.display_complete_duration = 2.0
        self.waiting_for_next = False 

    def load_templates(self, image_paths):
        for path in image_paths:
            template_image = cv2.imread(path)
            if template_image is None:
                raise ValueError(f"템플릿 이미지를 불러올 수 없습니다: {path}")
            
            template_image = cv2.resize(template_image, (640, 480))
            landmarks, pose_landmarks = self.extract_pose_landmarks(template_image)
            
            if landmarks is None:
                raise ValueError(f"템플릿 이미지에서 포즈를 감지할 수 없습니다: {path}")
            
            self.template_images.append(template_image.astype(np.float32) / 255.0)
            self.template_landmarks.append(landmarks)
            self.template_pose_landmarks.append(pose_landmarks)
            
    def reset_state(self):
        self.match_duration = 0
        self.blend_alpha = 0.0
        self.similarity = 0.0
        self.success_start_time = None 
        self.complete_time = None 
        self.completion_announced = False 
        
    def next_template(self):
        self.current_index += 1
        self.reset_state()
        return self.current_index < len(self.template_images)
    
    def extract_pose_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks), results.pose_landmarks
        return None, None

    def calculate_pose_similarity(self, landmarks1, landmarks2):
        """두 포즈 간의 유사도 계산 (포즈 + 위치 + 크기)"""
        if landmarks1 is None or landmarks2 is None:
            return 0.0

        # 1. 바운딩 박스 계산
        def get_pose_bounds(landmarks):
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]
            left = np.min(x_coords)
            right = np.max(x_coords)
            top = np.min(y_coords)
            bottom = np.max(y_coords)
            return left, right, top, bottom

        # 템플릿과 현재 포즈의 바운딩 박스 계산
        t_left, t_right, t_top, t_bottom = get_pose_bounds(landmarks1)
        c_left, c_right, c_top, c_bottom = get_pose_bounds(landmarks2)

        # 2. 크기 유사도 계산
        template_width = t_right - t_left
        template_height = t_bottom - t_top
        current_width = c_right - c_left
        current_height = c_bottom - c_top
        
        size_ratio_w = min(template_width, current_width) / max(template_width, current_width)
        size_ratio_h = min(template_height, current_height) / max(template_height, current_height)
        size_similarity = (size_ratio_w + size_ratio_h) / 2

        # 3. 위치 유사도 계산 (중심점 기준)
        template_center_x = (t_left + t_right) / 2
        template_center_y = (t_top + t_bottom) / 2
        current_center_x = (c_left + c_right) / 2
        current_center_y = (c_top + c_bottom) / 2

        position_diff_x = abs(template_center_x - current_center_x)
        position_diff_y = abs(template_center_y - current_center_y)
        position_similarity = 1 / (1 + position_diff_x + position_diff_y)

        # 4. 포즈 형태 유사도 계산
        landmarks1_norm = landmarks1 - np.mean(landmarks1, axis=0)
        landmarks2_norm = landmarks2 - np.mean(landmarks2, axis=0)

        scale1 = np.sqrt(np.sum(landmarks1_norm ** 2))
        scale2 = np.sqrt(np.sum(landmarks2_norm ** 2))
        
        landmarks1_scaled = landmarks1_norm / scale1
        landmarks2_scaled = landmarks2_norm / scale2

        pose_diff = landmarks1_scaled - landmarks2_scaled
        pose_distance = np.sqrt(np.sum(pose_diff ** 2))
        pose_similarity = 1 / (1 + pose_distance)

        # 5. 최종 유사도 계산
        total_similarity = (
            0.4 * pose_similarity +     # 포즈 형태
            0.3 * position_similarity + # 위치
            0.3 * size_similarity      # 크기
        )

        return total_similarity

    def create_body_mask(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_seg.process(image_rgb)
        
        if results.segmentation_mask is not None:
            mask = results.segmentation_mask
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            return mask
        return None

    def create_template(self, image_path):
        template_image = cv2.imread(image_path)
        if template_image is None:
            raise ValueError("템플릿 이미지를 불러올 수 없습니다.")
            
        template_image = cv2.resize(template_image, (640, 480))
        self.template_landmarks, self.template_pose_landmarks = self.extract_pose_landmarks(template_image)
        self.template_image = template_image.astype(np.float32) / 255.0
        
        if self.template_landmarks is None:
            raise ValueError("템플릿 이미지에서 포즈를 감지할 수 없습니다.")

    def apply_blending_effect(self, frame, similarity, pose_landmarks):
        frame_float = frame.astype(np.float32) / 255.0
        template_with_landmarks = self.template_images[self.current_index].copy()

        template_with_landmarks = (template_with_landmarks * 255).astype(np.uint8)
        if self.template_pose_landmarks[self.current_index]:
            self.mp_draw.draw_landmarks(
                template_with_landmarks,
                self.template_pose_landmarks[self.current_index],
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )
        template_with_landmarks = template_with_landmarks.astype(np.float32) / 255.0

        if self.complete_time is not None:
            if self.blend_alpha > 0:
                body_mask = self.create_body_mask(frame)
                if body_mask is not None:
                    mask3d = np.stack([body_mask] * 3, axis=2)
                    blended = template_with_landmarks.copy()
                    np.copyto(blended, frame_float, where=(mask3d > 0.1))
                    output = cv2.addWeighted(
                        template_with_landmarks, 0.1,
                        blended, 0.9,
                        0
                    )
                    return (output * 255).astype(np.uint8)
            return (template_with_landmarks * 255).astype(np.uint8)

        if similarity >= 0.85:
            if self.success_start_time is None:
                self.success_start_time = time.time()
            self.match_duration += 1
            self.blend_alpha = min(0.99, self.blend_alpha + 0.1)
        else:
            self.success_start_time = None
            self.match_duration = 0
            self.blend_alpha = max(0.0, self.blend_alpha - 0.05)

        if self.blend_alpha > 0:
            body_mask = self.create_body_mask(frame)
            if body_mask is not None:
                mask3d = np.stack([body_mask] * 3, axis=2)
                blended = template_with_landmarks.copy()
                np.copyto(blended, frame_float, where=(mask3d > 0.1))
                output = cv2.addWeighted(
                    template_with_landmarks, 0.1,
                    blended, 0.9,
                    0
                )

                if self.match_duration > 30:
                    glow = np.zeros_like(frame_float)
                    h, w = frame.shape[:2]
                    cv2.circle(glow, (w//2, h//2), 
                            int(min(w,h) * 0.4), 
                            (0.5, 1.0, 0.5), -1)
                    glow = cv2.GaussianBlur(glow, (21, 21), 0)
                    output = cv2.addWeighted(output, 1.0, glow, 0.2, 0)

                return (output * 255).astype(np.uint8)
        
        return (template_with_landmarks * 255).astype(np.uint8)

    def run_webcam_matching(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("웹캠에서 프레임을 읽을 수 없습니다.")
                    break
                
                frame = cv2.flip(frame, 1)
                
                current_landmarks, current_pose_landmarks = self.extract_pose_landmarks(frame)
                output_image = np.zeros((480, 1280, 3), dtype=np.uint8)

                if current_landmarks is not None and current_pose_landmarks is not None:
                    self.similarity = self.calculate_pose_similarity(
                        self.template_landmarks[self.current_index], 
                        current_landmarks
                    )

                    blended_template = self.apply_blending_effect(
                        frame, self.similarity, current_pose_landmarks
                    )
                    output_image[:, :640] = blended_template
                    
                else:
                    template_with_landmarks = self.apply_blending_effect(
                        frame, 0.0, None
                    )
                    output_image[:, :640] = template_with_landmarks

                output_image[:, 640:] = frame

                if current_pose_landmarks:
                    self.mp_draw.draw_landmarks(
                        output_image[:, 640:],
                        current_pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    )

                # 현재 포즈 번호 표시
                cv2.putText(output_image, f"Pose {self.current_index + 1}/{len(self.template_images)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

                score_text = f"Similarity: {self.similarity:.2%}"
                cv2.putText(output_image, score_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

                if self.success_start_time is not None:
                    elapsed_time = time.time() - self.success_start_time
                    if elapsed_time >= self.success_duration:
                        if self.complete_time is None:
                            self.complete_time = time.time()
                            if not self.completion_announced:
                                self.engine.say("Complete")
                                self.engine.runAndWait()
                                self.completion_announced = True
                                self.waiting_for_next = True
                        
                        # Complete 메시지 표시
                        cv2.putText(output_image, "Complete!",
                                (int(1280/2 - 150), int(480/2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 255, 0), 3)
                        
                        # Complete 메시지를 일정 시간 동안 표시
                        if self.waiting_for_next:
                            complete_elapsed = time.time() - self.complete_time
                            if complete_elapsed >= self.display_complete_duration:
                                self.waiting_for_next = False
                                if not self.next_template():
                                    cv2.putText(output_image, "All poses completed!",
                                            (int(1280/2 - 200), int(480/2) + 50),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1.5, (0, 255, 0), 3)
                                    cv2.imshow('Pose Matching', output_image)
                                    cv2.waitKey(2000)  # 2초 대기 후 종료
                                    break
                    else:
                        remaining = self.success_duration - elapsed_time
                        cv2.putText(output_image, f"Hold for {remaining:.1f}s",
                                (640 + 200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 255, 0), 3)

                cv2.putText(output_image, "Template Pose", (10, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_image, "Your Pose", (650, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_image, "Press 'q' to quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

                cv2.line(output_image, (640, 0), (640, 480), (255, 255, 255), 2)

                cv2.imshow('Pose Matching', output_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()
            self.selfie_seg.close()
            
def main():
    image_path = ["C:/Users/lmomj/Desktop/opensource/final/movies/dybala.jpg",
                  "C:/Users/lmomj/Desktop/opensource/final/movies/son.jpg",
                  "C:/Users/lmomj/Desktop/opensource/final/movies/yan.jpg"
                  ]
    
    try:
        system = PoseMatchingSystem()
        print("템플릿 포즈 추출 중...")
        system.load_templates(image_path)
        
        print("웹캠 매칭 시작...")
        system.run_webcam_matching()

    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == '__main__':
    main()