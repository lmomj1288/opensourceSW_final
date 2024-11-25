import cv2
import mediapipe as mp
import numpy as np

class SelfieSegmentation:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        # model_selection=1로 변경하여 더 정확한 모델 사용
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def process_image(self, image_path, bg_color=(192, 192, 192)):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("이미지를 불러올 수 없습니다.")

        # 이미지 전처리: 크기 조정 및 선명도 향상
        height, width = frame.shape[:2]
        # 이미지 크기를 조정하여 처리 성능 향상
        processed_frame = cv2.resize(frame, (width, height))
        # 언샤프 마스크로 이미지 선명도 향상
        gaussian = cv2.GaussianBlur(processed_frame, (0, 0), 2.0)
        processed_frame = cv2.addWeighted(processed_frame, 1.5, gaussian, -0.5, 0)

        # RGB 변환 및 세그멘테이션 수행
        RGB = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(RGB)

        if results.segmentation_mask is not None:
            # 마스크 후처리
            mask = results.segmentation_mask
            # 임계값 조정으로 더 선명한 경계 생성
            mask = np.where(mask > 0.2, 1, 0).astype(np.float32)
            
            # 모폴로지 연산으로 마스크 개선
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 가우시안 블러로 경계 부분 부드럽게 처리
            mask = cv2.GaussianBlur(mask, (7,7), 0)
            
            # 마스크 스택 및 조건 생성
            condition = np.stack((mask,) * 3, axis=-1) > 0.1

            # 배경 이미지 생성
            bg_image = np.zeros(processed_frame.shape, dtype=np.uint8)
            bg_image[:] = bg_color

            # 경계 부분 블렌딩을 위한 알파 블렌딩
            alpha = np.stack((mask,) * 3, axis=-1)
            output_image = cv2.convertScaleAbs(processed_frame * alpha + bg_image * (1 - alpha))

            # 마스크 이미지 생성 (시각화용)
            mask_image = np.uint8(mask * 255)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)

            return output_image, mask_image

        return frame, None

    def release(self):
        self.selfie_segmentation.close()

def main():
    # image_path = "C:/Users/lmomj/Desktop/opensource/project/sunflower.jpg"
    image_path = "C:/Users/lmomj/Desktop/opensource/final/movies/Jake_Sully.jpg"
    
    try:
        segmentation = SelfieSegmentation()
        output_image, mask_image = segmentation.process_image(image_path)

        # 결과 표시
        cv2.imshow('Original Image', cv2.imread(image_path))
        cv2.imshow('Segmentation Result', output_image)
        if mask_image is not None:
            cv2.imshow('Segmentation Mask', mask_image)

        # 결과 저장
        cv2.imwrite('improved_segmentation_result.jpg', output_image)
        if mask_image is not None:
            cv2.imwrite('improved_segmentation_mask.jpg', mask_image)

        cv2.waitKey(0)
        segmentation.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == '__main__':
    main()