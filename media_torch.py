import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image

class DeepLabSegmentation:
    def __init__(self):
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
        """
        마스크에서 edge를 추출하는 함수
        method: 'canny' 또는 'contour' 선택 가능
        thickness: edge의 두께
        """
        if method == 'canny':
            # Canny edge detection 사용
            edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
            # edge를 두껍게 만들기
            kernel = np.ones((thickness, thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # 빨간색 edge mask 생성
            colored_edges = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
            colored_edges[edges > 0] = [0, 0, 255]  # BGR 형식으로 빨간색 지정
            return colored_edges
        
        elif method == 'contour':
            # Contour 검출 사용
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            edge_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(edge_mask, contours, -1, (0, 0, 255), thickness)  # BGR 형식으로 빨간색 지정
            return edge_mask

    def process_image(self, image_path, bg_color=(192, 192, 192), edge_method='canny'):
        # 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0)
        
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
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

        # edge 추출 (빨간색)
        colored_edge_mask = self.get_edge_mask(mask, method=edge_method, thickness=3)

        # 원본 이미지를 numpy 배열로 변환
        original_image = np.array(image)

        # 원본 이미지와 edge 합성
        output_image = cv2.addWeighted(original_image, 1, colored_edge_mask, 1, 0)

        return output_image, colored_edge_mask

def main():
    image_path = "C:/Users/lmomj/Desktop/opensource/final/movies/ripley.jpg"
    
    try:
        segmentation = DeepLabSegmentation()
        output_image, edge_mask = segmentation.process_image(image_path, edge_method='canny')

        # 결과 표시
        cv2.imshow('Original Image', cv2.imread(image_path))
        cv2.imshow('Red Edge Mask', edge_mask)
        cv2.imshow('Image with Red Edge', output_image)

        print(edge_mask.shape)
        
        # edge_mask의 하단 절반만 저장
        h, w, chan = edge_mask.shape # (1381, 966, 3)
        # half_mask = edge_mask[h//2:, :, :]  # 하단 절반만 복사

        # 수정된 마스크 저장
        cv2.imwrite('deeplab_red_edge_mask.jpg', edge_mask)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == '__main__':
    main()