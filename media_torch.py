# mediapipe mp_selfie_segmentation 사용하는 게 아니라 resnet 기반 모델 사용 코드 

import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

class DeepLabSegmentation:
    def __init__(self):
        # DeepLab-V3+ 모델 로드 (사전 학습된 모델)
        # self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True) # computation issue로 inference가 너무 느림
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # 이미지 전처리를 위한 변환 정의
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image_path, bg_color=(192, 192, 192)):
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
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
        mask[output_predictions == 15] = 1  # 사람 클래스에 대한 마스크

        # 마스크 후처리
        mask = cv2.medianBlur(mask.astype(np.uint8), 7)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (7,7), 0)

        # 원본 이미지를 numpy 배열로 변환
        original_image = np.array(image)

        # 배경 이미지 생성
        bg_image = np.zeros_like(original_image)
        bg_image[:] = bg_color

        # 마스크를 3채널로 확장
        mask_3channel = np.stack([mask] * 3, axis=2)

        # 알파 블렌딩으로 이미지 합성
        output_image = cv2.convertScaleAbs(
            original_image * mask_3channel + bg_image * (1 - mask_3channel)
        )

        # 마스크 시각화를 위한 이미지 생성
        mask_visualization = (mask * 255).astype(np.uint8)
        mask_visualization = cv2.cvtColor(mask_visualization, cv2.COLOR_GRAY2BGR)

        return output_image, mask_visualization

def main():
    image_path = "C:/Users/lmomj/Desktop/opensource/project/titanic.jpg"  # 이미지 경로를 수정하세요
    
    try:
        segmentation = DeepLabSegmentation()
        output_image, mask_image = segmentation.process_image(image_path)

        # 결과 표시
        cv2.imshow('Original Image', cv2.imread(image_path))
        cv2.imshow('DeepLab Segmentation Result', output_image)
        cv2.imshow('Segmentation Mask', mask_image)

        # 결과 저장
        cv2.imwrite('deeplab_segmentation_result.jpg', output_image)
        cv2.imwrite('deeplab_segmentation_mask.jpg', mask_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == '__main__':
    main()