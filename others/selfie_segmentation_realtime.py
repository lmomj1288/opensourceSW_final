# real-time segmentation

import cv2
import mediapipe as mp
import numpy as np

class SelfieSegmentation:
    def __init__(self):
        # MediaPipe Selfie Segmentation 초기화
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def process_frame(self, frame, bg_color=(192, 192, 192)):
        # 프레임을 RGB로 변환
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Segmentation 수행
        results = self.selfie_segmentation.process(RGB)
        
        # 세그멘테이션 마스크가 있는 경우 처리
        if results.segmentation_mask is not None:
            # 마스크를 이용하여 배경 변경
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            
            # 배경 이미지 생성
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = bg_color
            
            # 원본 이미지와 배경 이미지를 마스크를 기반으로 합성
            output_image = np.where(condition, frame, bg_image)
            
            # 마스크 이미지 생성 (시각화용)
            mask_image = np.uint8(results.segmentation_mask * 255)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
            
            return output_image, mask_image
            
        return frame, None

    def release(self):
        self.selfie_segmentation.close()

def main():
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    segmentation = SelfieSegmentation()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 처리
        output_image, mask_image = segmentation.process_frame(frame)

        # 결과 표시
        cv2.imshow('Segmentation Result', output_image)
        if mask_image is not None:
            cv2.imshow('Segmentation Mask', mask_image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cap.release()
    segmentation.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()