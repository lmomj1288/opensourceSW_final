import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

class RealtimeDeepLabSegmentation:
    def __init__(self):
        # DeepLab-V3+ 모델 로드
        self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # 이미지 전처리를 위한 변환 정의
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def process_frame(self, frame, bg_color=(192, 192, 192)):
        # OpenCV BGR을 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # PIL Image로 변환
        image = Image.fromarray(frame_rgb)
        
        # 이미지 크기 조정 (성능 향상을 위해)
        width, height = 640, 480  # 원하는 크기로 조정 가능
        image = image.resize((width, height))
        
        # 모델 입력을 위한 전처리
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        # 클래스별 확률을 계산하고 가장 높은 확률의 클래스를 선택
        output_predictions = output.argmax(0).cpu().numpy()

        # 전경 마스크 생성 (사람 클래스 = 15)
        mask = np.zeros_like(output_predictions)
        mask[output_predictions == 15] = 1

        # 마스크 후처리
        mask = cv2.medianBlur(mask.astype(np.uint8), 7)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (7,7), 0)

        # 원본 프레임 크기 조정
        frame_resized = cv2.resize(frame, (width, height))

        # 배경 이미지 생성
        bg_image = np.zeros_like(frame_resized)
        bg_image[:] = bg_color

        # 마스크를 3채널로 확장
        mask_3channel = np.stack([mask] * 3, axis=2)

        # 알파 블렌딩으로 이미지 합성
        output_image = cv2.convertScaleAbs(
            frame_resized * mask_3channel + bg_image * (1 - mask_3channel)
        )

        return output_image

def main():
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    
    # 웹캠 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    segmentation = RealtimeDeepLabSegmentation()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break

            # 프레임 처리
            output_frame = segmentation.process_frame(frame)
            
            # 원본과 결과를 나란히 표시
            combined_frame = np.hstack((frame, output_frame))
            cv2.imshow('Realtime Segmentation (Original | Segmented)', combined_frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"에러 발생: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()