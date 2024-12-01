import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

class PoseTemplateSystem:
    def __init__(self):
        # DeepLab 모델 초기화
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
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
        self.mp_draw = mp.solutions.drawing_utils

    def get_edge_mask(self, mask, method='canny', thickness=5):
        """마스크에서 edge를 추출하는 함수"""
        if method == 'canny':
            edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
            kernel = np.ones((thickness, thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            colored_edges = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
            colored_edges[edges > 0] = [0, 0, 255]
            return colored_edges, edges  # edges도 함께 반환
        
        elif method == 'contour':
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            edge_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            binary_edge_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            cv2.drawContours(edge_mask, contours, -1, (0, 0, 255), thickness)
            cv2.drawContours(binary_edge_mask, contours, -1, 255, thickness)
            return edge_mask, binary_edge_mask

    def calculate_pose_match(self, frame, template_mask, template_binary):
        """포즈 매칭 점수 계산"""
        # MediaPipe를 사용하여 포즈 감지
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # 포즈 랜드마크를 이용해 마스크 생성
            h, w = frame.shape[:2]
            pose_mask = np.zeros((h, w), dtype=np.uint8)
            
            # 랜드마크 좌표를 이용해 마스크 생성
            landmarks = results.pose_landmarks.landmark
            points = []
            for landmark in landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points)
            cv2.fillPoly(pose_mask, [hull], 255)
            
            # 템플릿 마스크와 포즈 마스크의 겹치는 영역 계산
            intersection = cv2.bitwise_and(template_binary, pose_mask)
            template_area = cv2.countNonZero(template_binary)
            intersection_area = cv2.countNonZero(intersection)
            
            if template_area > 0:
                match_score = intersection_area / template_area
                return match_score, pose_mask
        
        return 0.0, None

    def create_template_mask(self, image_path):
        """이미지로부터 템플릿 마스크 생성"""
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
        edge_mask, binary_edge_mask = self.get_edge_mask(mask, method='canny', thickness=3)
        
        return edge_mask, binary_edge_mask

    def run_webcam_overlay(self, template_mask, template_binary):
        """웹캠에 템플릿 마스크 오버레이"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        template_height = int(480 * 0.8)
        template_width = int(640 * 0.8)
        template_mask = cv2.resize(template_mask, (640, template_height))
        template_binary = cv2.resize(template_binary, (640, template_height))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("웹캠에서 프레임을 읽을 수 없습니다.")
                    break

                frame = cv2.resize(frame, (640, 480))
                output_image = frame.copy()

                # 중앙 위치 계산
                start_y = (480 - template_height) // 2
                end_y = start_y + template_height

                # 포즈 매칭 점수 계산
                match_score, pose_mask = self.calculate_pose_match(
                    frame[start_y:end_y, :], 
                    template_mask,
                    template_binary
                )

                # 템플릿 마스크 오버레이
                mask_area = output_image[start_y:end_y, :]
                red_mask = template_mask[:, :, 2] > 0
                mask_area[red_mask] = mask_area[red_mask] * 0.3 + template_mask[red_mask] * 0.7

                # 매칭 점수 표시
                score_text = f"Match: {match_score:.2%}"
                cv2.putText(output_image, score_text, 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 95% 이상 매칭되면 1 출력
                if match_score >= 0.95:
                    cv2.putText(output_image, "1", 
                              (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)

                cv2.putText(output_image, "Press 'q' to quit", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Pose Template Overlay', output_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"에러 발생: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()

def main():
    image_path = "C:/Users/lmomj/Desktop/opensource/final/movies/faker.jpg"
    
    try:
        system = PoseTemplateSystem()
        
        print("템플릿 마스크 생성 중...")
        template_mask, template_binary = system.create_template_mask(image_path)
        
        print("웹캠 오버레이 시작...")
        system.run_webcam_overlay(template_mask, template_binary)

    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == '__main__':
    main()