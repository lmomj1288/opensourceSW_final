import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

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

    def get_edge_mask(self, mask, method='canny', thickness=5):
        """마스크에서 edge를 추출하는 함수"""
        if method == 'canny':
            edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
            kernel = np.ones((thickness, thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            colored_edges = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
            colored_edges[edges > 0] = [0, 0, 255]
            return colored_edges
        
        elif method == 'contour':
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            edge_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(edge_mask, contours, -1, (0, 0, 255), thickness)
            return edge_mask

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
        edge_mask = self.get_edge_mask(mask, method='canny', thickness=3)
        
        # 하단 절반만 추출
        h, w, _ = edge_mask.shape
        half_mask = edge_mask[h//2:, :, :]
        
        # return half_mask
        return edge_mask

    def run_webcam_overlay(self, template_mask):
        """웹캠에 템플릿 마스크 오버레이"""
        # 웹캠 초기화
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 템플릿 마스크 리사이징 (화면의 80% 크기로)
        template_height = int(480 * 0.8)
        template_mask = cv2.resize(template_mask, (640, template_height))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("웹캠에서 프레임을 읽을 수 없습니다.")
                    break

                # 프레임 크기 조정
                frame = cv2.resize(frame, (640, 480))
                
                # 결과 이미지 준비
                output_image = frame.copy()

                # 중앙 위치 계산
                start_y = (480 - template_height) // 2
                end_y = start_y + template_height

                # 템플릿 마스크를 중앙에 오버레이
                mask_area = output_image[start_y:end_y, :]
                
                # 마스크의 빨간색 부분만 오버레이
                red_mask = template_mask[:, :, 2] > 0
                mask_area[red_mask] = mask_area[red_mask] * 0.3 + template_mask[red_mask] * 0.7

                # 결과 표시
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

def main():
    # 이미지 경로 설정
    image_path = "C:/Users/lmomj/Desktop/opensource/final/movies/park.jpg"
    # image_path = "C:/Users/lmomj/Desktop/opensource/final/movies/ripleypython .jpg"    
    
    try:
        # 시스템 초기화
        system = PoseTemplateSystem()
        
        # 템플릿 마스크 생성
        print("템플릿 마스크 생성 중...")
        template_mask = system.create_template_mask(image_path)
        
        # 웹캠 오버레이 실행
        print("웹캠 오버레이 시작...")
        system.run_webcam_overlay(template_mask)

    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == '__main__':
    main()