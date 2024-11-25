import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import time 

class PoseTemplateSystem:
    def __init__(self):
        # DeepLab 모델 초기화
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 원본 포스터 저장 변수
        self.original_poster = None
        self.poster_size = None
        
        # 캡처 쿨다운을 위한 변수
        self.last_capture_time = 0
        self.capture_cooldown = 2.0  # 2초 쿨다운

    def get_edge_mask(self, mask, method='canny', thickness=5):
        """마스크에서 edge를 추출하는 함수"""
        if method == 'canny':
            edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
            kernel = np.ones((thickness, thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            colored_edges = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
            colored_edges[edges > 0] = [0, 0, 255]
            return colored_edges

    def create_template_mask(self, image_path):
        """이미지로부터 템플릿 마스크 생성"""
        # 원본 포스터 저장
        self.original_poster = cv2.imread(image_path)
        if self.original_poster is None:
            raise ValueError("포스터 이미지를 로드할 수 없습니다.")
        self.poster_size = self.original_poster.shape

        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        output_predictions = output.argmax(0).cpu().numpy()

        # 마스크 생성 및 후처리
        mask = np.zeros_like(output_predictions)
        mask[output_predictions == 15] = 1
        mask = cv2.medianBlur(mask.astype(np.uint8), 7)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (7,7), 0)

        # edge 추출
        edge_mask = self.get_edge_mask(mask, method='canny', thickness=3)
        
        # 하단 절반만 추출
        h, w, _ = edge_mask.shape
        half_mask = edge_mask[h//2:, :, :]
        
        return half_mask
        
    def create_pose_mask(self, frame, landmarks):
        """포즈 랜드마크로부터 마스크 생성"""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        points = []
        for landmark in landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.fillConvexPoly(mask, points, 255)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.GaussianBlur(mask, (11,11), 0)
        
        return mask

    def calculate_matching_score(self, pose_mask, template_area):
        """포즈 마스크와 템플릿 영역의 일치도 계산"""
        # 템플릿에서 빨간색 부분만 추출
        template_mask = template_area[:, :, 2] > 0
        template_mask = template_mask.astype(np.uint8) * 255
        
        # 포즈 마스크를 이진화
        pose_mask_bin = (pose_mask > 127).astype(np.uint8) * 255
        
        # 교집합 계산
        intersection = cv2.bitwise_and(pose_mask_bin, template_mask)
        intersection_area = np.count_nonzero(intersection)
        template_area = np.count_nonzero(template_mask)
        
        # 일치도 계산
        if template_area > 0:
            score = intersection_area / template_area
        else:
            score = 0
            
        return score

    def segment_person(self, frame):
        """프레임에서 사람 부분만 segmentation"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        output_predictions = output.argmax(0).cpu().numpy()
        
        mask = np.zeros_like(frame)
        mask[output_predictions == 15] = frame[output_predictions == 15]
        
        return mask

    def run_webcam_overlay(self, template_mask):
        """웹캠에 템플릿 마스크 오버레이 및 자동 캡처"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        template_height = int(480 * 0.8)
        template_mask = cv2.resize(template_mask, (640, template_height))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("웹캠에서 프레임을 읽을 수 없습니다.")
                    break

                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(frame_rgb)
                
                output_image = frame.copy()
                start_y = (480 - template_height) // 2
                end_y = start_y + template_height

                # 템플릿 마스크 오버레이
                mask_area = output_image[start_y:end_y, :]
                red_mask = template_mask[:, :, 2] > 0
                mask_area[red_mask] = mask_area[red_mask] * 0.3 + template_mask[red_mask] * 0.7

                if pose_results.pose_landmarks:
                    # 포즈 마스크 생성 및 매칭 점수 계산
                    pose_mask = self.create_pose_mask(frame[start_y:end_y, :], pose_results.pose_landmarks)
                    matching_score = self.calculate_matching_score(pose_mask, template_mask)
                    
                    # 매칭 점수 표시
                    cv2.putText(output_image, f"Match: {matching_score:.1%}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 90% 이상 일치하고 쿨다운이 지났으면 자동 캡처
                    current_time = time.time()
                    if matching_score >= 0.9 and (current_time - self.last_capture_time) >= self.capture_cooldown:
                        segmented_frame = self.segment_person(frame)
                        segmented_frame = cv2.resize(segmented_frame, (self.poster_size[1], self.poster_size[0]))
                        result = self.compose_with_poster(segmented_frame)
                        cv2.imwrite(f'composition_{int(current_time)}.jpg', result)
                        cv2.imshow('Captured Result', result)
                        print("자동 캡처 및 합성 완료!")
                        self.last_capture_time = current_time

                cv2.imshow('Pose Template Overlay', output_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"에러 발생: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def compose_with_poster(self, segmented_frame):
        """segmentation된 프레임을 원본 포스터와 합성"""
        mask = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2GRAY) > 0
        result = self.original_poster.copy()
        result[mask] = segmented_frame[mask]
        return result

def main():
    image_path = "C:/Users/lmomj/Desktop/opensource/final/movies/bakha.jpg"
    
    try:
        system = PoseTemplateSystem()
        
        print("템플릿 마스크 생성 중...")
        template_mask = system.create_template_mask(image_path)
        
        print("웹캠 오버레이 시작...")
        system.run_webcam_overlay(template_mask)

    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == '__main__':
    main()